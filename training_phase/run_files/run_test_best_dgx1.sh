#!/bin/bash

CODE=Multi_6layer_99_1_Bloss_mb2_ch12_64conv_V2loss_orilloss
SAVE_VISUAL=0
LR=0.5
GPU_ID=3
SUPERRES=0.0005
HIGHRES=0.9995
UNET_LEVEL=2
STAIN_NUM=5
TRAIN_LABEL_DIS="1_1_1_1_1_1_1_1_1_1"
MU_TIMES=1.0
SIGMA_TIMES=1.0
START_STAIN=0
GPU_ID_FOR_TRAINING=3
#TRAIN_LABEL_DIS="1_1"
# Name of the experiment
EXP_NAME=ByMZ-${CODE}-${UNET_LEVEL}-${LR}-${TRAIN_LABEL_DIS}-${SUPERRES}-${HIGHRES}-stain${STAIN_NUM}-mu${MU_TIMES}-sigma${SIGMA_TIMES}-start_stain${START_STAIN}-GPU${GPU_ID_FOR_TRAINING}

TEST_MODEL=/scratch/KurcGroup/mazhao/ICCV/output/3.21_multi_baseline/${EXP_NAME}/cnn_150.model
echo ${TEST_MODEL}
PRED_OUTPUT=../../output/${EXP_NAME}/pred_out_O0139_150
mkdir -p ${PRED_OUTPUT}

#TEST_CSV=/mnt/blobfuse/cnn-minibatches/lym_eval/label.csv
TEST_CSV=new_dots_test_O0135.csv
#maozheng_Multiplex_patch_list_test_2nd_batch.csv
#TEST_CSV=maozheng_Multiplex_patch_list_test_dots.csv

#maozheng_tumor_patch_list_testset_tcga_resized.csv
python -u test_model_multiplex_1stain_8layer_batchloss.py \
    --input=${TEST_CSV} \
    --output=${PRED_OUTPUT}/ \
    --model=${TEST_MODEL} \
    --gpuid=${GPU_ID} \
    --paral_code=0 \
    --paral_max=1 \
    --out_put_times=${UNET_LEVEL}\
    --stain_num=${STAIN_NUM}\
    > log_test_lym.txt

exit 0
