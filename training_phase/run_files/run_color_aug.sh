
CODE=ICCV_Train_with_dots_All_Data_newnet_5.1_backweight0.1_newBGR_aug_no_unknown_color-aug
SAVE_VISUAL=0
LR=0.1
GPU_ID=4
GPU_USED=4
SUPERRES=0.0005
HIGHRES=1.0
UNET_LEVEL=2
STAIN_NUM=7
TRAIN_LABEL_DIS="1_1_1_1_1_1_1_1_1_1"
MU_TIMES=1.0
SIGMA_TIMES=1.0
START_STAIN=0
TEST=1
# The distribution of data (in terms of NLCD labels) to sample from,
# in each training minibatch

#TRAIN_LABEL_DIS="1_1"
# Name of the experiment
EXP_NAME=ByMZ-${CODE}-${UNET_LEVEL}-${LR}-${TRAIN_LABEL_DIS}-${SUPERRES}-${HIGHRES}-stain${STAIN_NUM}-mu${MU_TIMES}-sigma${SIGMA_TIMES}-start_stain${START_STAIN}-GPU${GPU_USED}
echo ${EXP_NAME}
# The list of training and validation patches
#TRAIN_PATCH_LIST=/mnt/blobfuse/cnn-minibatches/lym_minipatch_pred_v0_list.txt
TRAIN_PATCH_LIST=../train_list_txt/maozheng_patch_list_multiplex.txt
TRAIN_PATCH_LIST_DOTS=../train_list_txt/maozheng_patch_list_multiplex_dots_no_unknown.txt
#maozheng_patch_list_multiplex_dots_dan.txt
#maozheng_patch_list_multiplex_dots_no_unknown.txt
TEST_PATCH_LIST=${TRAIN_PATCH_LIST}
# Output folder
# The models, copies of the code, log, etc. will be
# saved under ${OUTPUT}/${EXP_NAME}
# Check out /mnt/blobfuse/train-output/ for existing saved enrionments
OUTPUT=../../../output/DOTS_output/
mkdir -p ${OUTPUT}/${EXP_NAME}/
if [ ${TEST} -eq 0 ]; 
then
echo 'Training!!!!!!!'
cp -r ../*  ${OUTPUT}/${EXP_NAME}/
echo ${OUTPUT}/${EXP_NAME}/log.txt
python -u ../train_model.py \
    --name=${EXP_NAME} \
    --output=${OUTPUT}/ \
    --train_patch_list=${TRAIN_PATCH_LIST} \
    --train_patch_list_dots=${TRAIN_PATCH_LIST_DOTS} \
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
fi

#draw training loss
#python log2fig.py ${OUTPUT}/${EXP_NAME}

if [ ${TEST} -eq 1 ]
then

EPOCH=300

TEST_MODEL=${OUTPUT}/${EXP_NAME}/cnn_${EPOCH}.model
echo ${TEST_MODEL}

PRED_OUTPUT=${OUTPUT}/${EXP_NAME}/pred_out_O0135_${EPOCH}
mkdir -p ${PRED_OUTPUT}

TEST_CSV=../test_list_csv/new_dots_test_O0135.csv
#maozheng_Multiplex_patch_list_test_2nd_batch.csv

#maozheng_tumor_patch_list_testset_tcga_resized.csv
python -u ../test_model/test_model_multiplex_1stain_8layer_batchloss_no-softmax_multi_channel.py \
        --input=${TEST_CSV} \
        --output=${PRED_OUTPUT}/ \
        --model=${TEST_MODEL} \
        --gpuid=${GPU_ID} \
        --paral_code=0 \
        --paral_max=1 \
        --out_put_times=${UNET_LEVEL}\
        --stain_num=${STAIN_NUM}\
        --channel_times=12\
        > log_test_lym.txt

:'
TEST_MODEL=${OUTPUT}/${EXP_NAME}/cnn_${EPOCH}.model
echo ${TEST_MODEL}

PRED_OUTPUT=${OUTPUT}/${EXP_NAME}/pred_out_dots_${EPOCH}
mkdir -p ${PRED_OUTPUT}

TEST_CSV=new_dots_test_has_unknown.csv
#maozheng_Multiplex_patch_list_test_2nd_batch.csv
#maozheng_tumor_patch_list_testset_tcga_resized.csv
python -u test_model_multiplex_1stain_8layer_batchloss_dot_testset.py \
    --input=${TEST_CSV} \
    --output=${PRED_OUTPUT}/ \
    --model=${TEST_MODEL} \
    --gpuid=${GPU_ID} \
    --paral_code=0 \
    --paral_max=1 \
    --out_put_times=${UNET_LEVEL}\
    --stain_num=${STAIN_NUM}\
    > log_test_lym.txt
'

fi

exit 0

