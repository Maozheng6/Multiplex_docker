import os
import glob
#lists=os.listdir('/home/lehhou/maozheng/data/test_set/selected_tiles_for_labeling/')
lists=glob.glob('/scratch/KurcGroup/mazhao/ICCV/data_multiplex/dots_dan/O0135_N22800_regi_refined_4000-0.30_for_test/*wsi.png')
#('/home/lehhou/maozheng/data/test_set/TCGA_imgs_idx5_resized1.9076005961251865/')
print(len(lists))
f=open('O0135_N22800_wsi_ori_size_v3.0.csv','w')
f.write('patch_path,label'+'\n')
for i in lists:
    #f.write('/home/lehhou/maozheng/data/test_set/selected_tiles_for_labeling/'+i+',1'+'\n')
    if i.endswith('png'):
        f.write(i+',1'+'\n')
f.close()
