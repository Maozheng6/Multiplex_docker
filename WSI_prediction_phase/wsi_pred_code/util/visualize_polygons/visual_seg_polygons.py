from scipy import misc
from PIL import Image, ImageDraw
import numpy as np
import sys
import csv
import os
from write_polygons import write_polygons
seg_path = './116001_80001_4000_4000_0.25_1_SEG_argmax-binary.png'
#'/scratch/KurcGroup/mazhao/tiles_slide/O3936-multires/116001_80001_4000_4000_0.25_1_SEG.png'
out_path = './116001_80001_4000_4000_0.25_1_poly.png' # output file path

argmax_path ='/scratch/KurcGroup/mazhao/wsi_prediction/pred_out_iccv_resized_300/O3936_6.1/116001_80001_4000_4000_0.25_1_SEG_argmax.npy'
#'/scratch/KurcGroup/mazhao/quip4_poly_dots_model_resized/transfered10_300_no-hierar_argmax_maps/O3936-multires/116001_80001_4000_4000_0.25_1.npy'
poly_path = '/scratch/KurcGroup/mazhao/quip4_poly_dots_model_resized/transfered60_300_no-hierar/O3936-multires/K17_Neg/116001_80001_4000_4000_0.25_1-features.csv'
resize_ratio =1.0
#'./116001_80001_4000_4000_0.25_1-features.csv'
#'/scratch/KurcGroup/mazhao/quip4_poly_dots_model_resized/transfered10_300_no-hierar/O3936-multires/K17_Neg/116001_80001_4000_4000_0.25_1-features.csv'
#'./116001_80001_4000_4000_0.25_1-features.csv'

#write polygons#write polygons#write polygons
#write_polygons(argmax_path,6)

fields = os.path.basename(seg_path).split('_')
x_off = float(fields[0])
y_off = float(fields[1])
im = Image.open(seg_path).convert('RGB')
#im=im.resize((2000,2000))
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
        draw.line(tuple(coors), fill="red", width=2)
im.save(out_path)
im=im.resize((500,500))
im.save(out_path[0:-len('.png')]+'_thumb.png')


