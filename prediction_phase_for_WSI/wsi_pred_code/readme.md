# multiplex_wsi_pred
# 1_tiling_wsi_to_patches  

Go to ./multiplex_wsi_pred/1_tiling_wsi_to_patches

The WSI images are in /scratch/KurcGroup/mazhao/multiplex-wsi

To tile one WSI to patches, put the name of the slide in tiles_to_tile.txt, each row is a name of a WSI.

Then run "bash tile_all_tiles.sh", the log of the running prcess for each slide is saved in ./log_files.

The output patches are saved in ../../tiles_slide/

# 2_generate_patch_list_csv   

Go to ./multiplex_wsi_pred/2_generate_patch_list_csv

run "python name_list_for_test_csv.py <slide name>"
  
to generate the csv file which contains the list of names for patches, the csv file is saved in ./patch_lists_csv. And example of <slide name> is "N4277-multires.tif"

# 3_model_prediction_on_patches

Go to ./multiplex_wsi_pred/3_model_prediction_on_patches

Now we have two models, the model trained under 20x resolution and the model trained under 10x resolution. We predict the the results for each slide by twp models and combine the results after the prediction.

To run the prediction of 20x for one slide:

1) put the parameters in tiles_GPU.txt, and run 'bash run_all_tiles.sh'. Make sure the 'RESOLUTION=20x' in run_arg.sh.

Each row in tiles_GPU.txt is a process of running prediction.

<GPU to be used for this process> <slide name without '-multires.tif'> <maxmum number of processes run in parallel for this slide> <index of this process in the parallel> <suffix for this version model for differentiate the results from other models, for 20x model, the suffix should be the same>
  
For <maxmum number of processes run in parallel for this slide> <index of this process in the parallel>, we divide all the patches of a WSI into <maxmum number of processes run in parallel for this slide> parts, each process runs one part of the patches, the index of the part is <index of this process in the parallel>.

One example file content for tiles_GPU.txt is as follows:

7 N4277 3 0 _20x

7 N4277 3 1 _20x

7 N4277 3 2 _20x

It means using the 6th GPU to run the slide N4277-multires.tif, all the patches are divided into 3 parts, each process from 0-2 runs one of the part, the suffix for this version of results are _20x, all the results for this slide are saved in 'PRED_OUTPUT=../../wsi_pred_output/pred_out/${SLIDE}_${RESOLUTION}_${SUFFIX}' in run_arg.sh, Here it's '../../wsi_pred_output/pred_out/N4277_20x/' 

For running the 10x model, use 'RESOLUTION=10x' in run_arg.sh. One example of tiles_GPU_10x.txt is as follows:

6 N4277 3 0 _10x

6 N4277 3 1 _10x

6 N4277 3 2 _10x

The suffix of this model ("_10x") should be different from that of the 20x, so that the results are saved in different folders.

run "bash run_all_tiles.sh"

The results are in '../../wsi_pred_output/pred_out/N4277_10x/' .


# 3.5_merger_20x_10x

Change the lines in in run_merge.sh 

nohup python -u merge.py \
<10x results folder> \
<20x results folder> \
<folder to save the merged results from 10x and 20x results> &

For example,
nohup python -u merge.py \
/scratch/KurcGroup/mazhao/multiplex_docker/quip_ihc_analysis/Multiplex_seg_docker/wsi_pred/pred_out/3908_10x_pred/ \
/scratch/KurcGroup/mazhao/multiplex_docker/quip_ihc_analysis/Multiplex_seg_docker/wsi_pred/pred_out/3908_20x_pred/ \
/scratch/KurcGroup/mazhao/multiplex_docker/quip_ihc_analysis/Multiplex_seg_docker/wsi_pred/pred_out/3908_20x_10x_pred/ &
© 2020 GitHub, Inc.

Then run 'bash merge.sh'.

# 4_generate_polygons_and_meta_files_for_qui4

1) 1_run_poly_para_argmax.py

Change the following lines in the file to your own folders and suffix:

#####################################
#parameters to change
input_folder ='/scratch/KurcGroup/mazhao/multiplex_docker/quip_ihc_analysis/Multiplex_seg_docker/wsi_pred_output/pred_out/3908_10x_20x_pred'
pred_folder_name = os.path.basename(input_folder)
pred_folder_name_suffix = '_10x_20x_pred'
#'save_folder' is the output folder
save_folder='../../wsi_pred_output/json_csv/'

Then run:

python 1_run_poly_para_argmax.py 

This step generates the polygon files as .csv for each class and each patch, they are saved in 'save_folder'.

2)2_run_json.py 

Change the following lines in this file to your own folders and suffix:

######################################################
#input parameters
##################################################
#parameters to change
#'output_method_prefix' is the prefix of the method name shown on caMicroscope, for the combined results from 10x and 20x, it should be 'v8_10x20xcomb_'
output_method_prefix = 'v8_10x20x_comb_'

###################################################
#parameters that are same as the ones in 1_run_poly_para_argmax.py
input_folder ='/scratch/KurcGroup/mazhao/multiplex_docker/quip_ihc_analysis/Multiplex_seg_docker/wsi_pred_output/pred_out/3908_10x_20x_pred'
pred_folder_name = os.path.basename(input_folder)
input_folder_suffix = '_10x_20x_pred'
#'save_folder' is the output folder
save_folder='../../wsi_pred_output/json_csv/'

Then run:

python 2_run_json.py 

This step generates the meta data for each patch as .json files for uploading the polygons to quip4, the results are also saved in 'save_folder'. In that folder, for each patch there should be 2 files, one is .csv, the other is .json.
