import os
#lists=os.listdir('/home/lehhou/maozheng/data/test_set/selected_tiles_for_labeling/')
slide = ''
folder_name1= '/scratch/KurcGroup/mazhao/tiles_slide/single_stain/'
folder_name = folder_name1+slide
lists=os.listdir(folder_name)
#('/home/lehhou/maozheng/data/test_set/TCGA_imgs_idx5_resized1.9076005961251865/')
print(len(lists))
f=open(slide+'.csv','w')
f.write('patch_path,label'+'\n')
for i in lists:
    #f.write('/home/lehhou/maozheng/data/test_set/selected_tiles_for_labeling/'+i+',1'+'\n')
    if i.endswith('tif'):
        f.write(folder_name+i+',1'+'\n')
f.close()
