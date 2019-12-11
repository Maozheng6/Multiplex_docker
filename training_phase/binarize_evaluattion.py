import os
import cv2
import numpy as np
import copy
from skimage import morphology
import concurrent.futures
input_folder='./tumor_testset/'
files=os.listdir(input_folder)
save_folder='/mnt/blobfuse/train-output/ByMZ/bi_60_para_TCGA/binarized_prediction_without_holes_parallel/'
save_folder_edgeoverlay='/mnt/blobfuse/train-output/ByMZ/bi_60_para_TCGA/binarized_prediction_without_holes_parallel/edge_overlay/'
#GT_folder='/home/lehhou/maozheng/data/test_set/final_human_labels_v2/'
#testset='/home/lehhou/maozheng/data/test_set/selected_tiles_for_labeling/'
GT_folder='/home/lehhou/maozheng/data/test_set/TCGA_masks_resized/'
testset='/home/lehhou/maozheng/data/test_set/TCGA_imgs_idx5_resized/'
heat_from_classifier='/home/lehhou/maozheng/data/heat_from_classifier/'
remove_small=True
include_heat_from_classifier=False
def compute_IOU(GT,pred,mask=-1):

    #print('GT.shape,pred.shape',GT.shape,pred.shape)
    GT=(GT>0).astype('int')
    pred=(pred>0).astype('int')
    overlap=(GT*pred).astype('int')
    union=((GT+pred)>0).astype('int')
    if np.max(mask)<0:
        return np.sum(overlap),np.sum(union)
    else:
        mask=(mask>0).astype('int')
        return np.sum((overlap*mask).astype(int)),np.sum((union*mask).astype(int))

def band_mask(pred,GT,bandwidth):
    pred_edge=cv2.Canny(pred,100,200)
    #
    #cv2.imwrite('pred_edge.png',pred_edge)
    #
    GT_edge=cv2.Canny(GT,100,200)
    #cv2.imwrite('GT_edge.png',GT_edge)
    kernel=np.ones((bandwidth,bandwidth))
    pred_band=cv2.dilate(pred_edge,kernel,iterations=1)
    #
    #cv2.imwrite('pred_band.png',pred_band)
    #
    GT_band=cv2.dilate(GT_edge,kernel,iterations=1)
    #cv2.imwrite('GT_band.png',GT_band)
    band_combine=np.bitwise_or(pred_band,GT_band)
    #cv2.imwrite('band_combine.png',band_combine)
    #print(np.max(pred_band.astype('int')),np.max(GT_band.astype('int')))
    return band_combine

def edge(img,dim):
    img2=np.zeros((img.shape[0],img.shape[1],3)).astype('float')
    img1=cv2.Canny(img,100,200)
    kernel = np.ones((3,3),np.uint8)
    img1=cv2.dilate(img1,kernel,iterations = 1)

    for i in range(img2.shape[2]):
        if dim==12:
            img2[:,:,1]=img1.astype('float')
            img2[:,:,2]=img1.astype('float')
        else:
            if i==dim:
                img2[:,:,i]=img1.astype('float')
            else:
                img2[:,:,i]=-img1.astype('float')

    '''
    for i in img1.shape[2]:
        if i!=dim:
            img1[:,:,i]=0
    '''
    return img2

if not os.path.exists(save_folder):
    os.makedirs(save_folder)
if not os.path.exists(save_folder_edgeoverlay):
    os.makedirs(save_folder_edgeoverlay)


new_file_list=[]
for name in files:
    if (not (name.endswith('ori.png') or name.endswith('overlay.png'))) and name.endswith('.png'):
        if 1: #if not os.path.exists(save_folder_edgeoverlay+name.split('/')[-1][0:-4]+'_overlay_edge.png'):
            new_file_list.append(name)


def process_one_image(name):
    try:
        
        #initialize parameters
        heatoverlap=0
        heatunion=0
        overlap=0
        union=0
        overlap_masked=0
        union_masked=0
        overlap_masked_h=0
        union_masked_h=0
        
        heatmap=cv2.imread(os.path.join(input_folder,name),0)
        heatmap1,thre= cv2.threshold(heatmap,40,255,cv2.THRESH_BINARY)
        print('name',name,'thre.shape',thre.shape)
        cv2.imwrite(os.path.join(save_folder,name),thre)

        ori_img=cv2.imread(os.path.join(testset,name))

        pred=copy.deepcopy(thre)
        #print('pred.shape',pred.shape)
        ##################remove small obj holes
        if remove_small==True:
            pred=pred.astype('bool')
            pred= morphology.remove_small_objects(pred,4000, connectivity=2)
            pred=pred.astype('bool')
            pred= morphology.remove_small_holes(pred,4000, connectivity=2)
            pred=pred.astype('uint8')*255
            cv2.imwrite(os.path.join(save_folder,name),pred)
            
            ##########################
        '''
        thre[:,:,0]=0
        thre[:,:,2]=0
        overlay=np.clip(ori_img+(thre)*0.5,0,255)
        '''
        overlay=copy.deepcopy(ori_img)
        overlay[:,:,1]=np.clip(overlay[:,:,1]+(thre)*0.5,0,255)
        
        cv2.imwrite(save_folder+name.split('/')[-1][0:-4]+'_overlay.png',overlay)

        GT=cv2.imread(os.path.join(GT_folder,name),0)
        GT=cv2.resize(GT,(ori_img.shape[1],ori_img.shape[0]))
        #heat_from_classifier
        if include_heat_from_classifier==True:
            heatmap_from_classifyer=cv2.imread(os.path.join(heat_from_classifier,name),0)
            _,heatthre= cv2.threshold(heatmap_from_classifyer,242,255,cv2.THRESH_BINARY)
            cv2.imwrite(save_folder+name.split('/')[-1][0:-4]+'_heatbi.png',heatthre)
            heatoverlap,heatunion=compute_IOU(GT,heatthre)
            #overlap_list_h.append(heatoverlap)
            #union_list_h.append(heatunion)
        #IoU
        #print('1pred.shape',pred.shape)
        #print('1GT.shape',GT.shape)
        overlap,union=compute_IOU(GT,pred)

        #overlap_list.append(overlap)
        #union_list.append(union)
        
        #########################
        band_mask_thre=band_mask(pred,GT,int(240))
        overlap_masked,union_masked=compute_IOU(GT,pred,band_mask_thre)
        #overlap_masked_list.append(overlap_masked)
        #union_masked_list.append(union_masked)
        if include_heat_from_classifier==True:
            band_mask_heatthre=band_mask(heatthre,GT,int(240))
            overlap_masked_h,union_masked_h=compute_IOU(GT,heatthre,band_mask_heatthre)
            #overlap_masked_list_h.append(overlap_masked_h)
            #union_masked_list_h.append(union_masked_h)
        ##########################
        
        #
        edges_overlay=copy.deepcopy(ori_img)
        GT_edge=edge(GT,2)
        kernel = np.ones((5,5),np.uint8)
        GT_erode=cv2.erode(GT,kernel,iterations = 1)
        GT_erode_edge=edge(GT_erode,0)
        pred_edge=edge(pred,1)
        pred_erode=cv2.erode(pred,kernel,iterations = 1)
        pred_erode_edge=edge(pred_erode,0)
        #print('edge.shape',GT_edge.shape,np.max(GT_edge),ori_img.shape)
        if include_heat_from_classifier==True:
            heat_edge=edge(heatthre,12)
            kernel = np.ones((5,5),np.uint8)
            heat_erode=cv2.erode(heatthre,kernel,iterations = 1)
            heat_erode_edge=edge(heat_erode,0)
        if include_heat_from_classifier==True:
            edges_overlay=np.clip(edges_overlay+0.8*GT_erode_edge+0.8*pred_erode_edge+0.8*heat_erode_edge+1.0*heat_edge+1.0*GT_edge+1.0*pred_edge,0,255)
        else:
            edges_overlay=np.clip(edges_overlay+0.8*GT_erode_edge+0.8*pred_erode_edge+1.0*GT_edge+1.0*pred_edge,0,255)
        
        cv2.imwrite(save_folder_edgeoverlay+name.split('/')[-1][0:-4]+'_overlay_edge.png',edges_overlay)
        ########################
        ######mask visualize
        #########################
        edges_overlay=copy.deepcopy(ori_img)
        cv2.imwrite(save_folder_edgeoverlay+name.split('/')[-1][0:-4]+'band_mask_thre.png',band_mask_thre)
        edges_overlay[:,:,1]=np.clip(edges_overlay[:,:,1]+0.5*band_mask_thre.astype(int)*255,0,255)
        edges_overlay_GT_pred=np.clip(edges_overlay+0.8*GT_erode_edge+0.8*pred_erode_edge+1.0*GT_edge+1.0*pred_edge,0,255)
        cv2.imwrite(save_folder_edgeoverlay+name.split('/')[-1][0:-4]+'_overlay_edge_GT_pred_mask.png',edges_overlay_GT_pred)
        #####################
        if include_heat_from_classifier==True:
            edges_overlay=copy.deepcopy(ori_img)
            cv2.imwrite(save_folder_edgeoverlay+name.split('/')[-1][0:-4]+'band_mask_heatthre.png',band_mask_heatthre)
            edges_overlay[:,:,1]=np.clip(edges_overlay[:,:,1]+0.5*band_mask_heatthre.astype(int)*255,0,255)
            edges_overlay_GT_heat=np.clip(edges_overlay+0.8*GT_erode_edge+0.8*heat_erode_edge+1.0*GT_edge+1.0*heat_edge,0,255)
            cv2.imwrite(save_folder_edgeoverlay+name.split('/')[-1][0:-4]+'_overlay_edge_GT_heat_mask.png',edges_overlay_GT_heat)
        
        
        return (heatoverlap,heatunion,overlap,union,overlap_masked,union_masked,overlap_masked_h,union_masked_h)
    except:
        return None
        
        
        
def main():
    overlap_list=[]
    union_list=[]

    overlap_masked_list=[]
    union_masked_list=[]

    overlap_list_h=[]
    union_list_h=[]

    overlap_masked_list_h=[]
    union_masked_list_h=[]
    with concurrent.futures.ProcessPoolExecutor( max_workers=5) as executor:
        for number, prime in zip(new_file_list, executor.map(process_one_image, new_file_list, chunksize=5)):
            print('(heatoverlap,heatunion,overlap,union,overlap_masked,union_masked,overlap_masked_h,union_masked_h)')
            print('%s is prime: %s' % (number, prime))
            if prime==None:
                np.save(os.path.join(save_folder,'overlap_list'),overlap_list)
                np.save(os.path.join(save_folder,'union_list'),union_list)
                np.save(os.path.join(save_folder,'overlap_masked_list'),overlap_masked_list)
                np.save(os.path.join(save_folder,'union_masked_list'),union_masked_list)
                if include_heat_from_classifier==True:
                    np.save(os.path.join(save_folder,'overlap_list_h'),overlap_list_h)
                    np.save(os.path.join(save_folder,'union_list_h'),union_list_h)
                    np.save(os.path.join(save_folder,'overlap_masked_list_h'),overlap_masked_list_h)
                    np.save(os.path.join(save_folder,'union_masked_list_h'),union_masked_list_h)
                print('except happens at ', name)
                exit()
            else:
                (heatoverlap,heatunion,overlap,union,overlap_masked,union_masked,overlap_masked_h,union_masked_h)=prime
                overlap_list_h.append(heatoverlap)
                union_list_h.append(heatunion)
                overlap_list.append(overlap)
                union_list.append(union)
                overlap_masked_list.append(overlap_masked)
                union_masked_list.append(union_masked)
                overlap_masked_list_h.append(overlap_masked_h)
                union_masked_list_h.append(union_masked_h)
    
    #print(overlap_list,union_list)
    np.save(os.path.join(save_folder,'overlap_list'),overlap_list)
    np.save(os.path.join(save_folder,'union_list'),union_list)
    np.save(os.path.join(save_folder,'overlap_masked_list'),overlap_masked_list)
    np.save(os.path.join(save_folder,'union_masked_list'),union_masked_list)
    if include_heat_from_classifier==True:
        np.save(os.path.join(save_folder,'overlap_list_h'),overlap_list_h)
        np.save(os.path.join(save_folder,'union_list_h'),union_list_h)
        np.save(os.path.join(save_folder,'overlap_masked_list_h'),overlap_masked_list_h)
        np.save(os.path.join(save_folder,'union_masked_list_h'),union_masked_list_h)
    print('IoU is:',sum(overlap_list)/sum(union_list))
    print('DICE is:',2*sum(overlap_list)/(sum(union_list)+sum(overlap_list)))
    if include_heat_from_classifier==True:
        print('heat IoU is:',sum(overlap_list_h)/sum(union_list_h))

    print('masked IoU is:',sum(overlap_masked_list)/sum(union_masked_list))
    print('masked DICE is:',2*sum(overlap_masked_list)/(sum(union_masked_list)+sum(overlap_masked_list)))
    if include_heat_from_classifier==True:
        print('masked heat IoU is:',sum(overlap_masked_list_h)/sum(union_masked_list_h))

if __name__ == '__main__':
    main()
    #IoU=sum(overlap_list)/sum(union_list)
    


