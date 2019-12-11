# Multiplex_seg_16kdots
1)./run_files/run.sh will run training and testing process. The meaning of the parameters are as follows:

CODE=ICCV_Train_with_dots_new_Data_only_newnet_3.9_backweight0.1_newBGR_aug_nounknow
#CODE each version of results has a specific string of CODE to differentiate itself from other versions.

SAVE_VISUAL=0
#save visualization middle results (1) or not (0)

LR=0.1
#learning rate

GPU_ID=0
#The GPU id to run this test process.

GPU_USED=1
#The gpu id that the training process used for this version.

SUPERRES=0.0005
#weight of super resolution loss.

HIGHRES=1.0
#weight of high resolution loss.

UNET_LEVEL=2
#the output level of UNET, 1 is the highest resolution, 2 is half resolution, 3 is 1/4 res, 4 is 1/8 res.
#2 is the one we used.

STAIN_NUM=7
#number of stains in the model. 

TRAIN_LABEL_DIS="1_1_1_1_1_1_1_1_1_1"
#The string to define the sample rate of each class of the low resolution classes. There are low/high classes for 5 stains, so there are  10 classes. "1_1_1_1_1_1_1_1_1_1" means the same sample rate for each class.

MU_TIMES=1.0
#parameter to adjust the scale of the ground truth mu.

SIGMA_TIMES=1.0
#parameter to adjust the scale of the ground truth sigma.

START_STAIN=0
#from which stain to start, you can skip training some stain by setting this parameter to larger than 0.

TEST=0
#to run training (0) or test (1)


2)train_model.py is the pipeline of training the model.

3)MyDataSources.py is the dataloader part.

4)ModelLib.py defines the structure of the model.

5)./data/nlcd_mu.txt and ./data/nlcd_sigma.txt contain the ground truth mu and sigma.

6)./train_list_txt/ contains the txt files which have the lists of paths for training images. The ones we use now are  

maozheng_patch_list_multiplex.txt for the patches with low resolution labels, and

maozheng_patch_list_multiplex_dots_no_unknown.txt for the patches with high resolution labels.

7)./test_model/test_model_multiplex_1stain_8layer_batchloss_no-softmax_multi_channel.py is the code for testing the model on new images.

8)./eval_visual/visualize_dots_v5_bi_parall_supplementary_argmax.py is for visualizing the segmentation results on the testset and compute the F1 score for the testset.

