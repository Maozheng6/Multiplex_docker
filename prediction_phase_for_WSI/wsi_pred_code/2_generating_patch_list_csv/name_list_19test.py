import os
import sys
import glob
input_folder = '/scratch/KurcGroup/mazhao/multiplex_docker/quip_ihc_analysis/Multiplex_seg_docker/prediction_phase_for_WSI/wsi_pred_models/new_test_set'
#'/scratch/KurcGroup/mazhao/multiplex_docker/quip_ihc_analysis/Multiplex_seg_docker/wsi_pred/new_test_set/'
lists=glob.glob(input_folder+'/*png')
print(len(lists))
f=open('./patch_lists_csv/'+'19test.csv','w')
f.write('patch_path,label'+'\n')
for i in lists:
    f.write(i+',1'+'\n')
f.close()
