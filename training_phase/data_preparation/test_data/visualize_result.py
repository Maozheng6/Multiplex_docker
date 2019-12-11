from PIL import Image
import numpy as np
import glob
import os
import cv2
base = '/scratch/KurcGroup/mazhao/ICCV/data_multiplex/dots_dan/19test/'
subfolder = '19test_10x20x_v3'
file_i = 'N22800-multires_Image_985_ratio1.0_wsi_argmax.png'
image_path = os.path.join(base,'Image_985.png')
prediction_path =os.path.join(base,subfolder,file_i)
dots_map_path='/scratch/KurcGroup/mazhao/ICCV/data_multiplex/dots_dan/19test/Image_985_dots_map.png'
output_path=os.path.join(base,'visualize_pred_'+subfolder)
if not os.path.exists(output_path):
    os.makedirs(output_path)
def process_one(image_path,prediction_path,dots_map_path,output_path):
    GT=dots_map_path
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

    label = cv2.imread(GT,0)

    pred = cv2.imread(prediction_path,0)#np.load(prediction_path)

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

    #save ori
    #Image.fromarray(cv2.imread(image_path)[:,:,::-1]).save(os.path.join(output_path,os.path.basename(image_path)))
    #save pred only
    #Image.fromarray(pred_rgb).save(os.path.join(output_path,os.path.basename(image_path)[0:-len('.png')]+'_pred.png'))
    #save pred+dots
    #Image.fromarray(pred_lab_rgb).save(os.path.join(output_path,os.path.basename(image_path)[0:-len('.png')]+'_pred_label.png'))
    #save ori+label+dots
    Image.fromarray(np.concatenate((cv2.imread(image_path)[:,:,::-1], pred_lab_rgb), axis=0)).save(os.path.join(output_path,os.path.            basename(image_path)[0:-len('.png')]+'_im_pred_label.png'))

process_one(image_path,prediction_path,dots_map_path,output_path)

pred_files=glob.glob(os.path.join(base,subfolder)+'/*argmax.png')
for pred_i in pred_files:
    print('pred_i',pred_i)
    pred_split =os.path.basename(pred_i).split('_')
    image_name = pred_split[1]+'_'+ pred_split[2]+'.png'
    dots_map_name = pred_split[1]+'_'+ pred_split[2]+'_dots_map.png'
    print('dots_map_name',dots_map_name)
    image_path = os.path.join(base,image_name)
    prediction_path = pred_i
    dots_map_path = os.path.join(base,dots_map_name)
    process_one(image_path,prediction_path,dots_map_path,output_path)
