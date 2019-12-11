from scipy import misc
from PIL import Image, ImageDraw
import numpy as np
import sys
import csv
import os
from write_polygons import write_polygons
import pandas as pd

#folder having the plygons from step 4 for this slide, in this folder the subfolders are CD16*,CD3*,...
poly_folder = '/scratch/KurcGroup/mazhao/quip4_poly_dots_model_resized/6_slides_comp_v2/output_6_cropped_comp_rem-multires/'
save_folder = './overlayed_rm0_v2/'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
#The csv file from step 2, it saves the list of paths of all the patches
input_imgs_csv = '/scratch/KurcGroup/mazhao/wsi_pred_code/2_generating_patch_list_csv/patch_lists_csv/maozheng_Multiplex_patch_list_test_visual_comp.csv'
input_df = pd.read_csv(input_imgs_csv)
img_name_pair_list = input_df[["patch_path", "label"]].values
img_name_list = [xx[0] for xx in img_name_pair_list]


def visualize_one_image(img_path,img_suffix,poly_folder,poly_suffix,save_folder,save_suffix):
    #img_path: the path to the RGB image file
    #img_suffix: suffix of the img
    #poly_folder: folder of polygons, the subfolders of this folder are the 7 stains
    #poly_suffix: the suffix of the file names for the csv files
    #save_folder: folder to save the results
    #save_suffix: suffix of the saved output name
    img_prefix = os.path.basename(img_path).rstrip(img_suffix)
    cell_type={2:'CD3-Double_Negative_T_cell-Yellow',0:'CD16-Myeloid_cell-Black',4: 'CD8-Cytotoxic_cell-Purple',1:'CD20-B_cell-Red',3:'CD4-Helper_T_cell-Cyan',5:   'K17_Pos',6:'K17_Neg'}
    id2rgb = {
                #1: (55,247,61),
                1: (255,255,255),
                2: (13,240,18),
                3: (240,13,218),
                4: (0,255,255),
                5: (43,185,253),
                6: (227,137,26),
                7: (31,50,222),
                8: (255, 255, 255),
             }

    im = Image.open(img_path).convert('RGB')
    for stain_idx in cell_type:
        stain_folder = cell_type[stain_idx]

        poly_path =poly_folder+stain_folder+'/'+img_prefix+poly_suffix
        resize_ratio =1.0

        fields = os.path.basename(img_path).split('_')
        x_off = float(fields[0].split('--')[1])
        y_off = float(fields[1])
        draw = ImageDraw.Draw(im)

        with open(poly_path, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            line_count = 0
            for row in csv_reader:
                coors = [float(n) for n in row["Polygon"][1:-1].split(':')]
                for i in range(0, len(coors), 2):
                    coors[i] -= x_off
                for i in range(1, len(coors), 2):
                    coors[i] -= y_off
                coors += [coors[0], coors[1]]
                draw.line(tuple(coors), fill=id2rgb[stain_idx+1][::-1], width=3)
    out_path=os.path.join(save_folder,img_prefix+save_suffix)
    im.save(out_path)
    #im=im.resize((500,500))
    #im.save(out_path[0:-len('.png')]+'_thumb.png')

img_suffix = '.png'
poly_suffix = '_-features.csv'
save_suffix = '.png'
count =0
for img_path in img_name_list:
    count += 1
    print(img_path)
    print(float(count)/len(img_name_list))
    visualize_one_image(img_path,img_suffix,poly_folder,poly_suffix,save_folder,save_suffix)

