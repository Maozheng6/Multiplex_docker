from PIL import Image
import numpy as np
import glob
import os

base='/scratch/KurcGroup/mazhao/ICCV/output/DOTS_output/ByMZ-ICCV_Train_with_dots_new_Data_only_newnet_3.9_backweight0.1_newBGR_aug_nounknow-2-0.1-1_1_1_1_1_1_1_1_1_1-0.0005-1.0-stain7-mu1.0-sigma1.0-start_stain0-GPU1/pred_out_O0135_270/'
PRE=base+'Image_634_resized_{}.png'
save_folder=base+'/visual_folder_final/'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
def process_one(PRE):
    print(os.path.basename(PRE).split('_')[0])
    GT=glob.glob('/scratch/KurcGroup/mazhao/ICCV/data_multiplex/dots_dan/big_path_with_dots_testset_O0135/*{}*.npy'.format(os.path.basename(PRE).split('_')[1]))[0]
    print(GT)
    print('Found ground truth label {}'.format(GT))

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

    cd16 = np.array(Image.open(PRE.format(0)).convert('L'))[..., np.newaxis]
    cd20 = np.array(Image.open(PRE.format(1)).convert('L'))[..., np.newaxis]
    cd3  = np.array(Image.open(PRE.format(2)).convert('L'))[..., np.newaxis]
    cd4  = np.array(Image.open(PRE.format(3)).convert('L'))[..., np.newaxis]
    cd8  = np.array(Image.open(PRE.format(4)).convert('L'))[..., np.newaxis]
    k17  = np.array(Image.open(PRE.format(5)).convert('L'))[..., np.newaxis]
    k17n = np.array(Image.open(PRE.format(6)).convert('L'))[..., np.newaxis]
    bk   = np.array(Image.open(PRE.format(7)).convert('L'))[..., np.newaxis]
    label = np.load(GT)[..., -1]

    pred = np.concatenate((cd16, cd20, cd3, cd4, cd8, k17, k17n, bk), axis=-1)
    pred = np.argmax(pred, axis=-1) + 1

    pred_rgb = np.ones((pred.shape[0], pred.shape[1], 3), dtype=np.uint8) * 255
    for stainid in id2rgb.keys():
        print('Stain: {}, Count: {}'.format(stainid, (pred == stainid).sum()))
        pred_rgb[pred == stainid, :] = id2rgb[stainid]

    pred_lab_rgb = pred_rgb.copy()
    for stainid in id2rgb.keys():
        if stainid == 8:
            continue
        xs, ys = np.where(label == stainid)
        for x, y in zip(xs, ys):
            pred_lab_rgb[x-5:x+5, y-5:y+5] = id2rgb[stainid]
            invert_color = [255 - color for color in id2rgb[stainid]]
            pred_lab_rgb[x-5:x-5+1, y-5:y+5] = invert_color
            pred_lab_rgb[x+5:x+5+1, y-5:y+5] = invert_color
            pred_lab_rgb[x-5:x+5, y-5:y-5+1] = invert_color
            pred_lab_rgb[x-5:x+5, y+5:y+5+1] = invert_color

    #Image.fromarray(np.load(GT)[..., :3]).save(save_folder+os.path.basename(GT[0:-4])+'image.png')

    #Image.fromarray(pred_rgb).save(save_folder+GT[0:-4]+'pred.png')
    #Image.fromarray(pred_lab_rgb).save(save_folder+os.path.basename(GT[0:-4])+'pred_label.png')

    Image.fromarray(np.concatenate((np.load(GT)[..., :3], pred_lab_rgb), axis=0)).save(save_folder+os.path.basename(GT[0:-4])+'im_pred_label.png')

import glob
files=glob.glob('/scratch/KurcGroup/mazhao/ICCV/data_multiplex/dots_dan/big_path_with_dots_testset_O0135/*npy')

for file_i in  files:
        print(file_i)
        idx=os.path.basename(file_i).split('_')[1]
        print(idx)
        PRE=base+'Image_'+idx+'_resized_{}.png'
        process_one(PRE)
