#!/bin/bash
CODE=Multi_10layers_softmax_oriloss_nlcd_newdatav2_newmustd_background_count_1stain_1mu_sample
LR=0.0001
GPU_ID=3
SUPERRES=2
HIGHRES=1
UNET_LEVEL=16
# The distribution of data (in terms of NLCD labels) to sample from,
# in each training minibatch
TRAIN_LABEL_DIS="1_1"

# Name of the experiment
EXP_NAME=ByMZ-${CODE}-${UNET_LEVEL}-${LR}-${TRAIN_LABEL_DIS}-${SUPERRES}-${HIGHRES}

TEST_MODEL=/mnt/blobfuse/train-output/ByMZ/${EXP_NAME}/cnn_29.model
PRED_OUTPUT=/mnt/blobfuse/pred-output/${EXP_NAME}/
mkdir -p ${PRED_OUTPUT}
echo ${PRED_OUTPUT}

#TEST_CSV=/mnt/blobfuse/cnn-minibatches/lym_eval/label.csv
TEST_CSV=maozheng_Multiplex_patch_list_test.csv
#maozheng_tumor_patch_list_testset_tcga_resized.csv
python -u test_model_multiplex_le.py \
    --input=${TEST_CSV} \
    --output=${PRED_OUTPUT}/ \
    --model=${TEST_MODEL} \
    --gpuid=${GPU_ID} \
    --paral_code=0 \
    --paral_max=1 \
    --out_put_times=${UNET_LEVEL}\
    > log_test_multiplex_le.txt

exit 0
