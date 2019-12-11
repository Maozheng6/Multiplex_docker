import cv2
import numpy as np
import openslide
from PIL import Image

##################################
slide_name = 'L6745-multires'
wsi_path = '../../multiplex-wsi/'
low_res_size =np.array( [1335,787])
upper_left_in_low = np.array([359,597])
dan_tile_idx = 'Image_648'

#######################################
#whole slide size
oslide = openslide.OpenSlide(wsi_path+slide_name+'.tif')
oslide_size = np.array(oslide.level_dimensions[0])

upper_left_in_wsi =(oslide_size * (upper_left_in_low/low_res_size)).astype(int)
print(upper_left_in_wsi)
crop_size=(4000,4000)
cropped = oslide.read_region((upper_left_in_wsi[0],upper_left_in_wsi[1]), 0, crop_size)
resize_ratio=0.5*0.3468/0.293
new_size=(np.array(crop_size)*resize_ratio).astype(int)
cropped=cropped.resize(new_size)
save_folder = './cropped_imgs/'
cropped.save(save_folder+slide_name.split('-')[0]+'_wsi_'+dan_tile_idx+'_'+str(upper_left_in_low[0])+'_'+str(upper_left_in_low[1])+'_'+str(low_res_size[0])+'_'+str(low_res_size[1])+'.png')

cropped=cropped.resize((500,500))
cropped.save(save_folder+slide_name.split('-')[0]+'_wsi_'+dan_tile_idx+'_'+str(upper_left_in_low[0])+'_'+str(upper_left_in_low[1])+'_'+str(low_res_size[0])+'_'+str(low_res_size[1])+'_thumbnail.png')
