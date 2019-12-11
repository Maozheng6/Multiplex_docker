#!/bin/bash
CODE=Multi_6layer_99_1
SAVE_VISUAL=0
LR=0.1
GPU_ID=0
GPU_USED=1
SUPERRES=0.0005
HIGHRES=0.9995
UNET_LEVEL=2
STAIN_NUM=5
TRAIN_LABEL_DIS="1_1_1_1_1_1_1_1_1_1"
MU_TIMES=1.0
SIGMA_TIMES=1.0
START_STAIN=0
# The distribution of data (in terms of NLCD labels) to sample from,
# in each training minibatch

#TRAIN_LABEL_DIS="1_1"
# Name of the experiment
EXP_NAME=ByMZ-${CODE}-${UNET_LEVEL}-${LR}-${TRAIN_LABEL_DIS}-${SUPERRES}-${HIGHRES}-stain${STAIN_NUM}-mu${MU_TIMES}-sigma${SIGMA_TIMES}-start_stain${START_STAIN}-GPU${GPU_USED}
#/mnt/blobfuse/train-output/ByMZ-train_tumor_MZ1-0_1_1_1_1_1_1_1_1_1_1-2-40
# The list of training and validation patches
#TRAIN_PATCH_LIST=/mnt/blobfuse/cnn-minibatches/lym_minipatch_pred_v0_list.txt
TRAIN_PATCH_LIST=maozheng_patch_list_multiplex.txt
TEST_PATCH_LIST=${TRAIN_PATCH_LIST}

# Output folder
# The models, copies of the code, log, etc. will be
# saved under ${OUTPUT}/${EXP_NAME}
# Check out /mnt/blobfuse/train-output/ for existing saved enrionments
OUTPUT=/mnt/blobfuse/train-output/ByMZ/2.18/


#/mnt/blobfuse/train-output/ByMZ-train_tumor_MZ1-0_1_1_1_1_1_1_1_1_1_1-2-40
mkdir -p ${OUTPUT}/${EXP_NAME}/
cp -r ./data/ *.sh *.py ${OUTPUT}/${EXP_NAME}/
echo ${OUTPUT}/${EXP_NAME}/log.txt
:'
python -u train_model.py \
    --name=${EXP_NAME} \
    --output=${OUTPUT}/ \
    --train_patch_list=${TRAIN_PATCH_LIST} \
    --test_patch_list=${TEST_PATCH_LIST} \
    --gpuid=${GPU_ID} \
    --train_label_dis=${TRAIN_LABEL_DIS} \
    --superres=${SUPERRES} \
    --highres=${HIGHRES} \
    --unet_level=${UNET_LEVEL}\
    --initial_lr=${LR}\
    --stain_num=${STAIN_NUM}\
    --start_stain=${START_STAIN}\
    --mu_times=${MU_TIMES}\
    --sigma_times=${SIGMA_TIMES}\
    --save_visual=${SAVE_VISUAL}\
    2>&1 | tee log.txt > ${OUTPUT}/${EXP_NAME}/log.txt

'
#draw training loss
#python log2fig.py ${OUTPUT}/${EXP_NAME}
#EXP_NAME=ByMZ-${CODE}-${UNET_LEVEL}-${LR}-${TRAIN_LABEL_DIS}-${SUPERRES}-${HIGHRES}-stain${STAIN_NUM}-mu${MU_TIMES}-sigma${SIGMA_TIMES}-start_stain${START_STAIN}-GPU${GPU_ID_FOR_TRAINING}
TEST_MODEL=/mnt/blobfuse/train-output/ByMZ/2.18/${EXP_NAME}/cnn_10.model

echo ${TEST_MODEL}
echo ${TEST_MODEL}
PRED_OUTPUT=/mnt/blobfuse/train-output/ByMZ/2.18/pred_out/${EXP_NAME}/
mkdir -p ${PRED_OUTPUT}

#TEST_CSV=/mnt/blobfuse/cnn-minibatches/lym_eval/label.csv
TEST_CSV=maozheng_Multiplex_patch_list_test_dots.csv
#maozheng_tumor_patch_list_testset_tcga_resized.csv
python -u test_model_multiplex_1stain_6layer.py \
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
