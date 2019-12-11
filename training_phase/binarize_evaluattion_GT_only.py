import os
import cv2
import numpy as np
import copy
from skimage import morphology
input_folder='./tumor_testset/'
files=os.listdir(input_folder)
save_folder='./binarized_prediction_rmholes/'
save_folder_edgeoverlay='./binarized_prediction_rmholes/edge_overlay/'
GT_folder='/home/lehhou/maozheng/data/test_set/TCGA_masks_resized/'
testset='/home/lehhou/maozheng/data/test_set/TCGA_imgs_idx5_resized/'
heat_from_classifier='/home/lehhou/maozheng/data/heat_from_classifier/'
def compute_IOU(GT,pred):
    
    print('GT.shape',GT.shape)
    print('pred.shape',pred.shape)
    GT=(GT>0).astype('int')
    pred=(pred>0).astype('int')
    overlap=(GT*pred).astype('int')
    union=((GT+pred)>0).astype('int')
    return np.sum(overlap),np.sum(union)

    
def edge(img,dim):
    img2=np.zeros_like(img).astype('float')
    img1=cv2.Canny(img,100,200)
    kernel = np.ones((3,3),np.uint8)
    img1=cv2.dilate(img1,kernel,iterations = 1)
    
    for i in range(img2.shape[2]):
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
overlap_list=[]
union_list=[]

overlap_list_h=[]
union_list_h=[]
for name in files:
    try:
        if (not (name.endswith('ori.png') or name.endswith('overlay.png'))) and name.endswith('.png'):
            if not os.path.exists(save_folder_edgeoverlay+name.split('/')[-1][0:-4]+'_overlay_edge.png'):
                print(os.path.join(input_folder,name))
                print('here')
                heatmap=cv2.imread(os.path.join(input_folder,name))
                print('heatmap.shape',heatmap.shape)
                heatmap1,thre= cv2.threshold(heatmap,40,255,cv2.THRESH_BINARY)
                
                
                ori_img=cv2.imread(os.path.join(testset,name))

                pred=copy.deepcopy(thre)
                
                ##################remove small obj holes
                
                pred=pred.astype('bool')
                pred= morphology.remove_small_objects(pred,4000, connectivity=2)
                pred=pred.astype('bool')
                pred= morphology.remove_small_holes(pred,4000, connectivity=2)
                pred=pred.astype('uint8')*255
                cv2.imwrite(os.path.join(save_folder,name),pred)
                
                ##########################
                
                thre[:,:,0]=0
                thre[:,:,2]=0
                overlay=np.clip(ori_img+(thre)*0.5,0,255)
                cv2.imwrite(save_folder+name.split('/')[-1][0:-4]+'_overlay.png',overlay)
                
                GT=cv2.imread(os.path.join(GT_folder,name))
                GT=cv2.resize(GT,(pred.shape[1],pred.shape[0]))
                
                print(os.path.join(GT_folder,name),GT.shape)
                #heat_from_classifier
                '''
                heatmap_from_classifyer=cv2.imread(os.path.join(heat_from_classifier,name))
                
                _,heatthre= cv2.threshold(heatmap_from_classifyer,242,255,cv2.THRESH_BINARY)
                cv2.imwrite(save_folder+name.split('/')[-1][0:-4]+'_heatbi.png',heatthre)
                heatoverlap,heatunion=compute_IOU(GT,heatthre)
                overlap_list_h.append(heatoverlap)
                union_list_h.append(heatunion)
                '''
                #IoU
                print('pred.shape',pred.shape)
                overlap,union=compute_IOU(GT,pred)
                
                overlap_list.append(overlap)
                union_list.append(union)
                #
                edges_overlay=ori_img
                GT_edge=edge(GT,2)
                kernel = np.ones((5,5),np.uint8)
                GT_erode=cv2.erode(GT,kernel,iterations = 1)
                GT_erode_edge=edge(GT_erode,0)
                pred_edge=edge(pred,1)
                pred_erode=cv2.erode(pred,kernel,iterations = 1)
                pred_erode_edge=edge(pred_erode,0)
                #print('edge.shape',GT_edge.shape,np.max(GT_edge),ori_img.shape)
                edges_overlay=ori_img
                edges_overlay=np.clip(edges_overlay+0.8*GT_erode_edge+0.8*pred_erode_edge+1.0*GT_edge+1.0*pred_edge,0,255)
                #edges_overlay=np.clip(edges_overlay+1.0*GT_edge,0,255)
                
                #edges_overlay[:,:,1]=np.clip(edges_overlay[:,:,1]+pred_edge,0,255)
                
                #cv2.imwrite(save_folder+name.split('/')[-1][0:-4]+'_gt_edge.png',GT_edge)
                #cv2.imwrite(save_folder+name.split('/')[-1][0:-4]+'_pred_edge.png',pred_edge)
                cv2.imwrite(save_folder_edgeoverlay+name.split('/')[-1][0:-4]+'_overlay_edge.png',edges_overlay)
        except:
            np.save(os.path.join(save_folder,'overlap_list'),overlap_list)
            np.save(os.path.join(save_folder,'union_list'),union_list)
            print('except happens at ', name)
#IoU=sum(overlap_list)/sum(union_list)
print(overlap_list,union_list)
print('IoU is:',sum(overlap_list)/sum(union_list))
print('DICE is:',2*sum(overlap_list)/(sum(union_list)+sum(overlap_list)))
#print('heat IoU is:',sum(overlap_list_h)/sum(union_list_h))
        
        
        