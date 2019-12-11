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

