from PIL import Image
import numpy as np
import cv2
import os
import shutil
base='/scratch/KurcGroup/mazhao/ICCV/output/DOTS_output/ByMZ-ICCV_Train_with_dots_new_Data_only_newnet_3.9_backweight0.1_newBGR_aug_nounknow-2-0.1-1_1_1_1_1_1_1_1_1_1-0.0005-1.0-stain7-mu1.0-sigma1.0-start_stain0-GPU1/pred_out_O0135_270/'
imname='Image_642_resized.png'
shutil.copyfile(os.path.join(base,imname[:-4]+'_ori.png'),'./'+imname[:-4]+'_ori.png')
cell_type={'Yellow':'CD3 Double Negative T cell','Black':'CD16 Myeloid Cell','Purple':'CD8 Cytotoxic cell','Red':'CD20 B cell','Cyan':'CD4 helper T cell','K17+':'K17+','K17-':'K17-'}
heat_pred_stack = np.zeros((400,400,len(cell_type.keys())+1))
heat_layer_count=0



for color_num in range(len(cell_type.keys())+1):
    heat_pred=cv2.imread(os.path.join(base,os.path.basename(imname)[0:-4]+'_'+str(color_num)+'.png'),0)
    if heat_pred.shape[0]>400:
        heat_pred=cv2.resize(heat_pred,(400,400))
    heat_pred_stack[:,:,color_num] = heat_pred


print(heat_pred_stack.shape,np.max(heat_pred_stack))
arg=np.argmax(heat_pred_stack,axis=-1)+1
heat_pred_stack_bi=(heat_pred_stack>60).astype('uint8')
print(arg.shape,np.unique(arg))
cv2.imwrite('argmax.png',arg*40)



id2rgb = {
    1: (0, 0, 0),
    2: (255, 0, 0),
    3: (255, 255, 0),
    4: (10, 150, 255),
    5: (200, 0, 200),
    6: (100, 80, 80),
    7: (170, 170, 170),
    8: (255, 255, 255),
}

pred_rgb = np.ones((arg.shape[0], arg.shape[1], 3), dtype=np.uint8) * 255
for stainid in id2rgb.keys():
    print('Stain: {}, Count: {}'.format(stainid, (arg == stainid).sum()))
    pred_rgb[arg == stainid, :] = id2rgb[stainid]
Image.fromarray(pred_rgb).save('arg.png')



pred_rgb = np.ones((arg.shape[0], arg.shape[1], 3), dtype=np.uint8)

for i in range(7):
    temp = heat_pred_stack_bi[:,:,i]
    print('temp.shape',temp.shape)

    pred_rgb[temp==1, :] = id2rgb[i+1]

Image.fromarray(pred_rgb).save('thre.png')
