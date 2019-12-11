import os
#lists=os.listdir('/home/lehhou/maozheng/data/test_set/selected_tiles_for_labeling/')
input_folder = '/scratch/KurcGroup/mazhao/ICCV/data_multiplex/data_dots_labels_for_multiplex/2nd_batch/images/'
annotators = ['Areeha','Emily','Christian','Inga']
f=open('maozheng_Multiplex_patch_list_test_2nd_batch.csv','w')
f.write('patch_path,label'+'\n')

for ann in annotators:
    folder=os.path.join(input_folder,ann)
    lists=os.listdir(folder)
    #('/home/lehhou/maozheng/data/test_set/TCGA_imgs_idx5_resized1.9076005961251865/')
    print(len(lists))
    #f=open('maozheng_Multiplex_patch_list_test.csv','w')
    #f.write('patch_path,label'+'\n')
    for i in lists:
        if i.endswith('png'):
    #f.write('/home/lehhou/maozheng/data/test_set/selected_tiles_for_labeling/'+i+',1'+'\n')
            f.write(folder+'/'+i+',1'+'\n')
f.close()
