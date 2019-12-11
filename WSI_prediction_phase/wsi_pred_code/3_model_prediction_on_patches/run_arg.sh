GPU_ID=$1
RESOLUTION=20x
STAIN_NUM=7
UNET_LEVEL=2
SLIDE=$2
PARAL_MAX=$3
PARAL_CODE=$4
SUFFIX=$5
TEST_MODEL=../../wsi_pred_models/models/cnn_300_${RESOLUTION}.model
echo ${TEST_MODEL}

PRED_OUTPUT=../../wsi_pred_output/pred_out/${SLIDE}_${RESOLUTION}_${SUFFIX}
mkdir -p ${PRED_OUTPUT}

if [ ${RESOLUTION} == 20x ]
then
    RESIZE=0.5
    #0.844867
else
    RESIZE=0.25
    #0.422434
fi

TEST_CSV=../2_generating_patch_list_csv/patch_lists_csv/19test.csv
#3908-multires.csv
#19test.csv
#${SLIDE}-multires.csv
python -u test_model_multiplex_1stain_8layer_batchloss_no-softmax_nowhite_resize_fix-shuffle_argmax_visual_argmax-map_bgr-mode.py \
        --input=${TEST_CSV} \
        --output=${PRED_OUTPUT}/ \
        --model=${TEST_MODEL} \
        --gpuid=${GPU_ID} \
        --out_put_times=${UNET_LEVEL}\
        --stain_num=${STAIN_NUM}\
        --channel_times=12\
        --resize_ratio=${RESIZE}\
        --paral_code=${PARAL_CODE}\
        --paral_max=${PARAL_MAX}\
        --input_image_mode='BGR'        
        > log_test_lym_GPU${GPU_ID}_${SLIDE}.txt

exit 0

