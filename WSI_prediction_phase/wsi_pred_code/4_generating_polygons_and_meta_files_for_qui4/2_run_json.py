import os
import glob
from gen_json import start_json
import sys
######################################################
#input parameters
##################################################
#parameters to change
#'output_method_prefix' is the prefix of the method name shown on caMicroscope, for the combined results from 10x and 20x, it should be 'v8_10x20xcomb_'
output_method_prefix = 'v8_10x20x_comb_'

###################################################
#parameters that are same as the ones in 1_run_poly_para_argmax.py
input_folder ='/scratch/KurcGroup/mazhao/multiplex_docker/quip_ihc_analysis/Multiplex_seg_docker/wsi_pred_output/pred_out/3908_20x_pred'
pred_folder_name = os.path.basename(input_folder)
input_folder_suffix = '_20x_pred'
#'save_folder' is the output folder
save_folder='../../wsi_pred_output/json_csv/'

######################################################
#parameters not to change
slide_idx = pred_folder_name[0:-len(input_folder_suffix)]#pred_folder_name. split('_')[0]

output_folder_suffix = '-multires'

input_file_suffix = '_1_SEG_argmax.png'
output_file_suffix = '-algmeta.json'

input_file_suffix_for_search = 'argmax.png'

slide_suffix = '-multires.tif'
stain_num=7
###############################################
#start computing

folders=os.listdir(os.path.dirname(input_folder))

cell_type={2:'CD3-Double_Negative_T_cell-Yellow',0:'CD16-Myeloid_cell-Black',4:'CD8-Cytotoxic_cell-Purple',1:'CD20-B_cell-Red',3:'CD4-Helper_T_cell-Cyan',5:'K17_Pos',6:'K17_Neg'}

for slide_i in folders:
    print('slide_i',slide_i)
    #debug!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if not (slide_i.endswith(slide_idx+input_folder_suffix)):
        continue
    for stain_idx in range(stain_num):
        png_path= os.path.dirname(input_folder) +'/'+ slide_i+'/*'+input_file_suffix_for_search
        print('png_path',png_path)
        print(glob.glob(png_path))
        inpath = os.path.join(input_folder,slide_i)
        #slide_idx = slide_i.rstrip(input_folder_suffix)
        save_sub_folder=os.path.join(save_folder,slide_idx +output_folder_suffix,cell_type[stain_idx])
        print('slide_idx',slide_idx)
        print('output_folder_suffix',output_folder_suffix)
        analysis_id = output_method_prefix+cell_type[stain_idx]
        print('slide_i,stain_idx',slide_i,stain_idx)
        start_json(slide_i,stain_idx,inpath,save_sub_folder,output_method_prefix,analysis_id,output_folder_suffix,png_path,input_file_suffix,output_file_suffix,slide_suffix,input_folder_suffix)
