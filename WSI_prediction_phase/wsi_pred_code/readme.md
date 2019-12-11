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

1) put the parameters in tiles_GPU_20x.txt, and run 'bash run_all_tiles_20x.sh'.

Each row in tiles_GPU_20x.txt is a process of running prediction.

<GPU to be used for this process> <slide name without '-multires.tif'> <maxmum number of processes run in parallel for this slide> <index of this process in the parallel> <suffix for this version model for differentiate the results from other models, for 20x model, the suffix should be the same>
  
For <maxmum number of processes run in parallel for this slide> <index of this process in the parallel>, we divide all the patches of a WSI into <maxmum number of processes run in parallel for this slide> parts, each process runs one part of the patches, the index of the part is <index of this process in the parallel>.

One example file content for tiles_GPU_20x.txt is as follows:

7 N4277 3 0 _6.1_1.0

7 N4277 3 1 _6.1_1.0

7 N4277 3 2 _6.1_1.0

It means using the 6th GPU to run the slide N4277-multires.tif, all the patches are divided into 3 parts, each process from 0-2 runs one of the part, the suffix for this version of results are _6.6_1.0, all the results for this slide are saved in /scratch/KurcGroup/mazhao/wsi_prediction/pred_out_6_slides_300/N4277_6.1_1.0

For running the 10x model, one example of tiles_GPU_10x.txt is as follows:

6 N4277 3 0 _6.6_1.0

6 N4277 3 1 _6.6_1.0

6 N4277 3 2 _6.6_1.0

The suffix of this model ("_6.6_1.0") should be different from that of the 20x, so that the results are saved in different folders.

run "bash run_all_tiles_10x.sh"

The results are in /scratch/KurcGroup/mazhao/wsi_prediction/pred_out_6_slides_300/N4277_6.6_1.0


# 3_model_prediction_on_patches_shahira

This part is for predicting with pytorch model.

Change the following to make the code predicting with pytorch model.


In run_arg_20x.sh change the TEST_MODEL to the directory of your pytorch model.

In test_model_multiplex_1stain_8layer_batchloss_no-softmax_nowhite_resize_fix-shuffle_argmax_visual_argmax-map_bgr-mode.py, in lines with 'model.eval' are predicting with the cntk models, change this line to pytorch model prediction.

model = cntk.load_model(model_fn) is loading the cntk model, it should be changed to pytorch model.


input_shape = model.arguments[0].shape is getting the imput image shape of the network.



# 4_generate_polygons_and_meta_files_for_qui4

1)python 1_run_poly_para_argmax.py <slide name> <suffix indicating the version of the model> 

For example,

python 1_run_poly_para_argmax.py N4277-multires.tif _6.6_1.0

This step generates the polygon files as .csv for each class and each patch, they are saved in '../../quip4_poly_dots_model_resized/6_slides/'

2)python 2_run_json.py <slide name> <suffix indicating the version of the model> <prefix indicating this version of polygon on quip4>

For example:

python 2_run_json.py 3908-multires.tif _6.6_1.0 v8

This step generates the meta data for each patch as .json files for uploading the polygons to quip4, the results are also saved in '../../quip4_poly_dots_model_resized/6_slides/'. In that folder, for each patch there should be 2 files, one is .csv, the other is .json.