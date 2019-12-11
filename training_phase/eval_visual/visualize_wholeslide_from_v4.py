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
binarize_thre=int(sys.argv[2])
results=sys.argv[1]
print('results',results)
files=glob.glob(input_folder+'/*.zip')


base_folder='../../data_multiplex/data_dots_labels_for_multiplex/2nd_batch'
unzipped_folder='unzipped'#Areeha/2938_cd20h_cd3h_cd4h_cd8h.png-points/'
image_folder='images'#Areeha/2938_cd20h_cd3h_cd4h_cd8h.png'
save_folder='dots_visualization_with_seg_16_deconve_dgx1_bionly'#Areeha/'
annotator='Inga'
def visualize_one_patch(imname,save_folder):
    color_idx={'Yellow':2, 'Purple':4,'Black':0, 'Cyan':3, 'Red':1}
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    #load dots
    print(imname)
    image_name=imname.split('/')[-1]
    cell_type={'Yellow':'CD3 Double Negative T cell','Black':'CD16 Myeloid Cell','Purple':'CD8 Cytotoxic cell','Red':'CD20 B cell','Cyan':'CD4 helper T cell'}
    for color in cell_type.keys():
        heat_pred=cv2.imread(os.path.join(results,os.path.basename(imname)[0:-4]+'_'+str(color_idx[color])+'.png'))
        if heat_pred.shape[0]>400:
            heat_pred=cv2.resize(heat_pred,(400,400))
        heatmap1,heat_binary= cv2.threshold(heat_pred,binarize_thre,255,cv2.THRESH_BINARY)
        cv2.imwrite(os.path.join(save_folder,os.path.basename(imname)[0:-4]+'_'+color+'.png'),heat_binary)
    return 0


annotators=['Areeha','Christian','Emily','Inga']
colors_by_annotators={}
#for annotator in annotators:
def process_one_patch_parall(imname):
    precision_dict={'Black':[],'Red':[],'Yellow':[],'Cyan':[],'Purple':[]}
    recall_dict={'Black':[],'Red':[],'Yellow':[],'Cyan':[],'Purple':[]}
    colors_by_one_annotator = []
    annotator = imname.split('/')[-2]
    dots_folder = os.path.join(base_folder,unzipped_folder,annotator,os.path.basename(imname)+'-points')
    save_folder_processed_dots = os.path.join(base_folder,'processed_dots',annotator)
    #print(dots_folder,os.path.exists(dots_folder))
    save_dir = os.path.join(results,save_folder,annotator)
    #print('save_dir',save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    precision_dict1={}
    recall_dict1={}
    visualize_one_patch(imname,save_dir)
    return 0
def main():
    with concurrent.futures.ProcessPoolExecutor( max_workers=40) as executor:
        for number, pr in zip(im_names, executor.map(process_one_patch_parall, im_names, chunksize=2)):
            print('%s is prime: %s' % (number, pr))

    return 0


if __name__ == '__main__':

    im_names=[]
    for annotator in ['Christian','Emily']:#['Areeha','Christian','Emily', 'Inga']:# ['Christian','Emily']:# ['Areeha','Christian','Emily','Inga']:
       im_names+=glob.glob(os.path.join(base_folder,image_folder,annotator)+'/*.png')
    #im_names+=glob.glob(os.path.join(base_folder,image_folder,'Emily')+'/2271_cd16h_cd4h_cd8.png')
    main()

