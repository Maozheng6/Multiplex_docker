import os
import time
import glob
import concurrent.futures
import cv2
from get_poly import get_poly
import numpy as np
import sys
#######################################################
#input parameters
# For example:
# If the input folder name is O3936_6.6_1.0, the output save_folder name will   be O3936-multires
# If the input file name is 60001_40001_4000_4000_0.25_1_SEG_argmax.npy, the    output file name will be 60001_40001_4000_4000_0.25-features.csv
# The suffix in the following are for change the input folder/file names to the output name.
#####################################
#parameters to change
input_folder ='/scratch/KurcGroup/mazhao/multiplex_docker/quip_ihc_analysis/Multiplex_seg_docker/wsi_pred_output/pred_out/3908_20x_pred'
pred_folder_name = os.path.basename(input_folder)
pred_folder_name_suffix = '_20x_pred'
#'save_folder' is the output folder
save_folder='../../wsi_pred_output/json_csv/'

######################################################
#parameters not to change
slide_idx = pred_folder_name[0:-len(pred_folder_name_suffix)]#pred_folder_name.split('_')[0]
print('pred_folder_name',pred_folder_name)
print('slide_idx',slide_idx)
input_folder_suffix = pred_folder_name[len(slide_idx):]
output_folder_suffix = '-multires'

input_file_suffix ='_1_SEG_argmax.png'
output_file_suffix = '-features.csv'

#'input_file_suffix_for_search' is for searching all the input files.
input_file_suffix_for_search = 'argmax.png'
infolder =input_folder

#########################################################################
#start computing
folders = [infolder]
cell_type_with_BG={2:'CD3-Double_Negative_T_cell-Yellow',0:'CD16-Myeloid_cell-Black',4:'CD8-Cytotoxic_cell-Purple',1:'CD20-B_cell-Red',3:'CD4-Helper_T_cell-Cyan',5:'K17_Pos',6:'K17_Neg',7:'BG'}

cell_type={2:'CD3-Double_Negative_T_cell-Yellow',0:'CD16-Myeloid_cell-Black',4:'CD8-Cytotoxic_cell-Purple',1:'CD20-B_cell-Red',3:'CD4-Helper_T_cell-Cyan',5:'K17_Pos',6:'K17_Neg'}


make_list = []

folder_i = infolder
folder_i_base = os.path.basename(folder_i)
print('folder_i_base',folder_i_base)

files = glob.glob(os.path.join(folder_i,'*'+input_file_suffix_for_search))
for file_i in files:
    make_list.append([file_i,folder_i])
print('make_list',make_list)

class make_pair_list():
    def __init__(self):
        self.pair_list=[]

    def make_argmax_and_list_for_one(self,pair):
        if True:

            file_i,folder_i = pair
            argmax_name = os.path.join(folder_i,os.path.basename(file_i))

            print('argmax_name',argmax_name)

            for stain in cell_type.keys():
                save_path = os.path.join(save_folder,os.path.basename(folder_i)[0:-len(input_folder_suffix)]+output_folder_suffix,cell_type[stain])


                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                file_prefix=os.path.basename(file_i)[0:-len(input_file_suffix)]
                if  not os.path.isfile(os.path.join(save_path,file_prefix+output_file_suffix)):
                    print(os.path.isfile(os.path.join(save_path,file_prefix+output_file_suffix)))
                    print(os.path.join(save_path,file_prefix+'-features.csv'))
                    self.pair_list.append([file_i,save_path,stain,argmax_name,input_file_suffix,output_file_suffix])
                    print('len(self.pair_list)',len(self.pair_list))
                else:
                    print(os.path.join(save_path,file_prefix+'-features.csv'),'exists______________')
                    print(os.path.isfile(os.path.join(save_path,file_prefix+output_file_suffix)))

def main():
    MAKE_PAIR=make_pair_list()
    #with concurrent.futures.ProcessPoolExecutor( max_workers=10) as executor:
    #    for  prime in  executor.map(MAKE_PAIR.make_argmax_and_list_for_one, make_list, chunksize=10):
    #        print(' is prime: %s' % ( prime))

    print('len(make_list)',len(make_list))
    for pp in make_list:
        MAKE_PAIR.make_argmax_and_list_for_one(pp)
    print('len(pair_list)',len(MAKE_PAIR.pair_list))
    print('len(make_list)',len(make_list))
    with concurrent.futures.ProcessPoolExecutor( max_workers=30) as executor:
        for number, prime in zip(MAKE_PAIR.pair_list, executor.map(get_poly, MAKE_PAIR.pair_list, chunksize=4)):
            print('%s is prime: %s' % (number, prime))


def main_none_para():
    #fid=open(save_folder+'/'+'rm_log.txt','w')
    for pair in pair_list:
        a=get_poly(pair)
if __name__ == '__main__':
    main()
    #main_none_para()
