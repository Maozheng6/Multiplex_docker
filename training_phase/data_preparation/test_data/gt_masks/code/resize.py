import glob
import os
import cv2
files = glob.glob('/scratch/KurcGroup/mazhao/ICCV/micro1/Multiplex_seg_v6_ICCV_rebuttal+instance_cell_8.19/data_preparation/test_data/gt_masks/data/mask/*png')
save_path = '/scratch/KurcGroup/mazhao/ICCV/micro1/Multiplex_seg_v6_ICCV_rebuttal+instance_cell_8.19/data_preparation/test_data/gt_masks/data/mask400'
label_path = '/scratch/KurcGroup/mazhao/ICCV/micro1/Multiplex_seg_v6_ICCV_rebuttal+instance_cell_8.19/data_preparation/test_data/gt_masks/data/label400'
label_channel_path = '/scratch/KurcGroup/mazhao/ICCV/micro1/Multiplex_seg_v6_ICCV_rebuttal+instance_cell_8.19/data_preparation/test_data/gt_masks/data/label_channel400/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
    print(os.path.exists(save_path))

if not os.path.exists(label_path):
        os.makedirs(label_path)
if not os.path.exists(label_channel_path):
        os.makedirs(label_channel_path)

label2color = {1:(255, 255, 255),
                       2:(13, 230, 18),
                                      3:(240, 13, 128),
                                                     4:(255, 211, 0),
                                                                    5:(43,185,253),
                                                                                   6:(227, 137, 26)
                                                                                           }
import numpy as np
for file_i in files:
    img = cv2.imread(file_i)
    img =  img[0:474,0:474,:]
    img = cv2.resize(img,(400,400),cv2.INTER_NEAREST)
    print('img.shape',img.shape)
    #####################
    mask = img
    range_s=10
    label_channel = np.zeros((400,400,6))
    for stain in range(1,7):
        this_stain1 = (mask[:,:,0] >= label2color[stain][2]-range_s) &          (mask[:,:,0] <= label2color[stain][2]+range_s)

        print('this_stain1',np.max(this_stain1))
        this_stain2 = (label2color[stain][1]-range_s<=mask[:,:,1]) & (mask[:,:, 1] <=label2color[stain][1]+range_s)
        print('this_stain2',np.max(this_stain2))

        this_stain3 = (label2color[stain][0]-range_s<= mask[:,:,2]) & (mask[:,:,2]<=label2color[stain][0]+range_s)
        this_stain = this_stain3 & this_stain2 &this_stain1
        this_stain = this_stain.astype('uint8')
        cv2.imwrite(os.path.join(label_path,os.path.basename(file_i)[0:-len('.png')])+'_'+str(stain)+'.png',this_stain*255)
        label_channel[:,:,stain-1] = this_stain#cv2.resize(this_stain,(400,400),cv2.INTER_NEAREST)

    ###########
    np.save(label_channel_path+os.path.basename(file_i)[0:-len('.png')],label_channel)
    #####################
    img_re = cv2.resize(img,(400,400))#,interpolation = cv2.INTER_NEAREST)
    print(os.path.join(save_path,os.path.basename(file_i)))
    cv2.imwrite(os.path.join(save_path,os.path.basename(file_i)),img_re)


