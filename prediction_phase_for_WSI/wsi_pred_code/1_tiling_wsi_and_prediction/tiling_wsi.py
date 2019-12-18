import numpy as np
import openslide
import sys
import os
from PIL import Image
import cv2
def stain_normalized_tiling(outfolder,slide_name, patch_size, do_actually_read_image=True):
    margin = 5
    try:
        oslide = openslide.OpenSlide(slide_name)
        if openslide.PROPERTY_NAME_MPP_X not in oslide.properties:
            mpp = 0.25
        else:
            mpp = float(oslide.properties[openslide.PROPERTY_NAME_MPP_X])
        if mpp < 0.375:
            scale_factor = 1
        else:
            scale_factor = 2
        pw = patch_size
        width = oslide.dimensions[0]
        height = oslide.dimensions[1]
    except:
        print('Error in {}: exception caught exiting'.format(slide_name))
        raise Exception('{}: exception caught exiting'.format(slide_name))
        return

    #n40X = reinhard_normalizer('color_norm/target_40X.png')

    for x in reversed(range(1, width, pw)):
        for y in reversed(range(1, height, pw)):
            if x + pw > width - margin:
                pw_x = width - x - margin
            else:
                pw_x = pw
            if y + pw > height - margin:
                pw_y = height - y - margin
            else:
                pw_y = pw

            if pw_x <= 0 or pw_y <= 0:
                continue

            outf = os.path.join(outfolder, '{}_{}_{}_{}_{}_{}_SEG.png'.format(x, y, pw_x, pw_y,   mpp, scale_factor))
            exist_flag = 0
            if os.path.isfile(outf):
                exist_flag=1
                print('exists:',outf)
            if do_actually_read_image and exist_flag==0:
                try:
                    patch = oslide.read_region((x, y), 0, (pw_x, pw_y)).convert('RGB')
                except:
                    print('{}: exception caught'.format(slide_name))
                    continue
            elif exist_flag==0:
                patch = Image.new('RGB', (pw_x, pw_y), (255, 255, 255))
            else:
                #place holder
                patch = Image.new('RGB', (10, 10), (255, 255, 255))
            ori_size0 = patch.size[0]
            ori_size1 = patch.size[1]
            patch = np.array(patch.resize(
                (patch.size[0]*scale_factor, patch.size[1]*scale_factor), Image.ANTIALIAS))

            yield patch, (exist_flag,x, y, pw_x, pw_y, ori_size0, ori_size1, mpp, scale_factor), (width, height)

#ii=stain_normalized_tiling('/scratch/KurcGroup/mazhao/multiplex-wsi/O3936-multires.tif',40, True)
#print(next(ii)[1])
#print(next(ii)[1])
out_path='../../../wsi_output/'

def cnn_pred_mask( wsi_path):
    PS =800
    step_size = 400
    gsm = np.ones((PS, PS, 1), dtype=np.float32) * 1e-6
    gsm[1:-1, 1:-1, 0] = 0.01;
    gsm[50:-50, 50:-50, 0] = 1;
    batch_size = 10;

    outfolder = os.path.join(out_path, os.path.basename(wsi_path))
    if not os.path.isdir(outfolder):
        os.makedirs(outfolder)
    do_gpu_process = True
    tiling= stain_normalized_tiling(wsi_path, 4000, do_gpu_process)
    while tiling:
        try:
            uint8patch, patch_info, wsi_dim = next(tiling)
            print(patch_info)
            #patch = uint8patch.astype(np.float32)/255
            px, py, pw_x, pw_y, ori_size0, ori_size1, mpp, scale_factor = patch_info
            outf = os.path.join(outfolder, '{}_{}_{}_{}_{}_{}_SEG.png'.format(
                                         px, py, pw_x, pw_y, mpp, scale_factor))
            # Check if patch is too small to handle
            if patch.shape[0] < PS or patch.shape[1] < PS:
                continue;
            # Check if skip the CNN step
            if do_gpu_process:
                print("CNN segmentation on", outf)
                pred_m = np.zeros((patch.shape[0], patch.shape[1], 3), dtype=np.float32);
                num_m = np.zeros((patch.shape[0], patch.shape[1], 1), dtype=np.float32) + 4e-6;
                net_inputs = [];
                xy_indices = [];
                for x in list(range(0, pred_m.shape[0]-PS+1, step_size)) + [pred_m.shape[0]-PS,]:
                    for y in list(range(0, pred_m.shape[1]-PS+1, step_size)) + [pred_m.shape[1]-PS,]:
                        pat = patch[x:x+PS, y:y+PS, :]
                        wh = pat[...,0].std() + pat[...,1].std() + pat[...,2].std();
                        if 1:#wh >= 0.01:#0.18:
                            net_inputs.append(pat.transpose());
                            xy_indices.append((x, y));
                            print(x,y)
                            outf_1 = os.path.join(outfolder, '{}_{}_{}_{}_{}_{}_{}_{}_SEG.png'.format(
                                         px, py, pw_x, pw_y, mpp, scale_factor,x,y))
                            #cv2.imwrite(outf_1, pat);
                            pat_pil = Image.fromarray(pat, 'RGB')
                            pat_pil.save(outf_1)

                        '''
                        if len(net_inputs)>=batch_size or (x==pred_m.shape[0]-PS and y==pred_m.shape[1]-PS and len(net_inputs)>0):
                            feed_dict = {
                              self.model.test_patch_normalized: np.transpose(np.array(net_inputs), [0,2,3,1]),
                            }
                            res_discrim = self.model.test_learner_patch(self.sess, feed_dict, None, with_output=True);
                            Inet_outputs = res_discrim['output']
                            net_outputs = np.concatenate((net_outputs, np.zeros((
                                net_outputs.shape[0], net_outputs.shape[1], net_outputs.shape[2], 1), dtype=np.float32)), axis=-1)
                            net_outputs = np.swapaxes(net_outputs, 1, 2)
                            for outi, (x, y) in enumerate(xy_indices):
                                pred_m[x:x+PS, y:y+PS, :] += net_outputs[outi, ...] * gsm;
                                num_m[x:x+PS, y:y+PS, :] += gsm;
                            net_inputs = [];
                            xy_indices = [];
                        '''

                '''
                pred_m /= num_m;
                pred_m = misc.imresize((pred_m*255).astype(np.uint8), (ori_size0, ori_size1));
                imwrite(outf, pred_m);
                '''

        except StopIteration:
            break
    return


def load_model(model_fn):

    ######################
    #model_fn,
     # Make sure the input files are OK and ensure output directory exist

    if not os.path.isfile(model_fn):
        logging.error("Model file does not exist, exiting")
        return


    model = cntk.load_model(model_fn)
    output_shape = model.outputs[0].shape
    input_shape = model.arguments[0].shape
    model_input_size = input_shape[1]

    return model,output_shape,input_shape,model_input_size

def predict_one_patch(naip_tile,resize_ratio,channel_times,model,model_input_size,out_put_times,stain_num,savefolder,  naip_fn, skip_exists):
    #################################
    #resize to correct size
    print('resizing input')
    try:
        input_ori_shape = naip_tile.shape
    except AttributeError:
        print('files does not exists:',naip_fn)
        return None,None,None,None

    naip_tile = cv2.resize(naip_tile,(int(naip_tile.shape[1]*resize_ratio),int(naip_tile.shape[0]*resize_ratio)))
    print('input size after resize',naip_tile.shape)
    #############################
    # if the predction file exists, skip predcition preocess
    if skip_exists ==True  and  os.path.exists(savefolder+os.path.basename(naip_fn)[0:-4]+'_argmax.png'):
        print('skip_exists',skip_exists)
        print('File exits, skip this file:',savefolder+os.path.basename(naip_fn)[0:-4]+'_argmax.png')
        #return None
    else:
        print('----------------------------compute this file:',savefolder+os.path.basename(naip_fn)[0:-4]+'_argmax.    png')
    ##############################
    ##############################
    #skip this image if it's white image
    white_image=False
    if (naip_tile.std()<6 and naip_tile.mean()>215) or (naip_tile.shape[0]<400 or naip_tile.shape[1]<400):
        white_image=True
        print('White_image!!!')
    naip_tile=naip_tile/255.0
    #naip_tile=naip_tile[:,:,::-1]
    for che in range(channel_times):
        if che == 0:
            naip_im_channels = naip_tile
        else:
            naip_im_channels = np.concatenate(              [naip_im_channels,naip_tile],axis=2)
    naip_tile=naip_im_channels
    naip_tile=np.swapaxes(naip_tile,2,0)
    naip_tile=np.swapaxes(naip_tile,2,1)
    naip_tile=np.squeeze(naip_tile)
    edges_overlay=np.array((copy.deepcopy(naip_tile)*255).astype('uint8'))
    edges_overlay=np.swapaxes(edges_overlay,2,0)
    edges_overlay=np.swapaxes(edges_overlay,0,1)
    edges_overlay=edges_overlay[:,:,0:3]
    ori_img=edges_overlay
    if white_image!=True :
        output = run_model_on_tile(model, naip_tile.astype(np.float32), model_input_size, 32,out_put_times,stain_num)
    else:
        height = naip_tile.shape[1]
        width = naip_tile.shape[2]
        output =  np.zeros((height,width, stain_num+1), dtype=np.float32)
        output[:,:,-1] = 1.0
    height = naip_tile.shape[1]
    width = naip_tile.shape[2]
    print('output.shape',output.shape)
    #stain_pred=[]
    #overlayed=[]
    #expand output
    #output_expand = np.expand_dims(output,0)
    #out_expand = expand_output_mao(output_expand,1/resize_ratio)
    #out_expand = out_expand[0,:,:,:]
    out_expand = cv2.resize(output,(input_ori_shape[1],input_ori_shape[0]))
    print('out_expand.shape',out_expand.shape)
    argmax_map = np.argmax(out_expand,axis=2)+1
    argmax_name = savefolder+os.path.basename(naip_fn)[0:-4]+'_argmax.png'
    #np.save(argmax_name,argmax_map)
    cv2.imwrite(argmax_name,argmax_map.astype('uint8'))
    return 1,argmax_map,output,ori_img


#slide_list=['O6218_MULTI_3-multires.tif','3908-multires.tif','O3936-multires.tif', 'L6745-multires.tif', 'O3105-multires.tif', 'O0135-multires.tif', 'N9430-multires.tif', 'N22034-multires.tif']
wsi_path = sys.argv[1]
slide=os.path.basename(wsi_path)
#slide_list[0]
print('slide to tile',slide)
#cnn_pred_mask( wsi_path)
outfolder='../../../tiles_slide/'+slide[:-4]+'/'
if not os.path.exists(outfolder):
    os.makedirs(outfolder)
tiling= stain_normalized_tiling(outfolder, wsi_path, 4000, True)
while tiling:
    uint8patch, patch_info, wsi_dim = next(tiling)
    exist_flag, px, py, pw_x, pw_y, ori_size0, ori_size1, mpp, scale_factor = patch_info
    print(px,py)
    outf = os.path.join(outfolder, '{}_{}_{}_{}_{}_{}_SEG.png'.format(px, py, pw_x, pw_y, mpp, scale_factor))
    if exist_flag ==0:
        uint8patch = Image.fromarray(uint8patch)
        uint8patch.save(outf)

