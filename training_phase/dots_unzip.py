import os
#results='/mnt/blobfuse/train-output/ByMZ/2.18/pred_out/ByMZ-Multi_6layer_99_1_withBN_yb99.5_yerode1_cyan.75_mustdv_2-2-0.1-1_1_1_1_1_1_1_1_1_1-0.0005-0.9995-stain5-mu1.0-sigma1.0-start_stain0-GPU0/'
input_folder='/mnt/blobfuse/train-output/ByMZ/data_dots_labels_for_multiplex/zipped/Inga/Inga_Completed/Inga_Completed_Annot/'
import glob
import shutil
import cv2
import copy
import matplotlib.pyplot as plt
import concurrent.futures
import numpy as np
import collections
import pickle
import sys
import json




base_folder='/mnt/blobfuse/train-output/ByMZ/data_dots_labels_for_multiplex/2nd_batch/'
unzipped_folder='unzipped'#Areeha/2938_cd20h_cd3h_cd4h_cd8h.png-points/'
image_folder='images'#Areeha/2938_cd20h_cd3h_cd4h_cd8h.png'
save_folder='dots_visualization_with_seg_16'#Areeha/'
annotator='Christian'

files=glob.glob(os.path.join(base_folder,annotator)+'/*.zip')
#unzip
for file_i in files:
    if file_i.endswith('.zip'):
        print('unzip '+file_i+' '+os.path.join(base_folder,unzipped_folder,annotator)+'/'+os.path.basename(file_i)[0:-4])
        if not os.path.exists(os.path.join(base_folder,unzipped_folder,annotator)+'/'+os.path.basename(file_i)[0:-4]):
            os.makedirs(os.path.join(base_folder,unzipped_folder,annotator)+'/'+os.path.basename(file_i)[0:-4])
        shutil.copyfile(file_i,os.path.join(os.path.join(base_folder,unzipped_folder,annotator),os.path.basename(file_i)))
        os.system('unzip '+os.path.join(base_folder,unzipped_folder,annotator)+'/'+os.path.basename(file_i))
        os.system('cp *.txt '+os.path.join(base_folder,unzipped_folder,annotator)+'/'+os.path.basename(file_i)[0:-4])
        os.system('rm *.txt')
