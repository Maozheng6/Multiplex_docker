import os
lists=os.listdir('/scratch/KurcGroup/mazhao/ICCV/data_multiplex/multiplex10_patches_labelled/multiplex_training_with_background_label/')
#('/mnt/blobfuse/cnn-minibatches/maozheng_data/maozheng_tumor_region_ratio_240')
print(len(lists))
f=open('maozheng_patch_list_multiplex.txt','w')
for i in lists:
    if i.endswith('.npy'):
        f.write('/scratch/KurcGroup/mazhao/ICCV/data_multiplex/multiplex10_patches_labelled/multiplex_training_with_background_label/'+i+'\n')
        #('/mnt/blobfuse/cnn-minibatches/maozheng_data/maozheng_tumor_region_ratio_240/'+i+'\n')

f.close()
