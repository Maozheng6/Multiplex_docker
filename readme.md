
To run the docker, run the following line:

docker run --name test8 --rm -it -v {wsi_pred_out}:/root/Multiplex_docker/wsi_pred_out -v {WSI_folder}:/root/Multiplex_docker/WSI maozheng_test /bin/bash 

where {wsi_pred_out} is the local folder to save the WSI prediction results.
{WSI} is the folder with whole-slide images.

The instructions for the prediction phase are in ./prediction_phase_for_WSI/wsi_pred_code/
The instructions for the training phase are in ./training_phase/
