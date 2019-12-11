#!/bin/bash

CODE=Multi_6layer_99_1_withBN_yb99.5_yerode1_cyan.75_mustdv3
SAVE_VISUAL=0
LR=0.5
GPU_ID=1
SUPERRES=0.0005
HIGHRES=0.9995
UNET_LEVEL=2
STAIN_NUM=5
TRAIN_LABEL_DIS="1_1_1_1_1_1_1_1_1_1"
MU_TIMES=1.0
SIGMA_TIMES=1.0
START_STAIN=0
GPU_ID_FOR_TRAINING=0
#TRAIN_LABEL_DIS="1_1"
# Name of the experiment
EXP_NAME=ByMZ-${CODE}-${UNET_LEVEL}-${LR}-${TRAIN_LABEL_DIS}-${SUPERRES}-${HIGHRES}-stain${STAIN_NUM}-mu${MU_TIMES}-sigma${SIGMA_TIMES}-start_stain${START_STAIN}-GPU${GPU_ID_FOR_TRAINING}

TEST_MODEL=/mnt/blobfuse/train-output/ByMZ/2.18/${EXP_NAME}/cnn_20.model
echo ${TEST_MODEL}
PRED_OUTPUT=/mnt/blobfuse/train-output/ByMZ/2.18/${EXP_NAME}/pred_out_deconve
mkdir -p ${PRED_OUTPUT}

#TEST_CSV=/mnt/blobfuse/cnn-minibatches/lym_eval/label.csv
TEST_CSV=maozheng_Multiplex_patch_list_test_dots_2nd_batch.csv
#maozheng_tumor_patch_list_testset_tcga_resized.csv
python -u test_model_multiplex_1stain_6layer_L2baseline.py \
    --input=${TEST_CSV} \
    --output=${PRED_OUTPUT}/ \
    --model=${TEST_MODEL} \
    --gpuid=${GPU_ID} \
    --paral_code=0 \
    --paral_max=1 \
    --out_put_times=1\
    --stain_num=${STAIN_NUM}\
    > log_test_lym.txt

exit 0
