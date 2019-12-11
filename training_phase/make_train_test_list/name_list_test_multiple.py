import os
#lists=os.listdir('/home/lehhou/maozheng/data/test_set/selected_tiles_for_labeling/')
slides_list=['N3908_CD16_BLACK_10-multires',  'N3908_K17_BROWN_9-multires','O6218_CD8_PURPLE_8-multires', 'N3908_CD20_RED_6-multires','O6218_CD16_BLACK_10-multires','O6218_K17_BROWN_9-multires','N3908_CD3_YELLOW_5-multires','O6218_CD20_RED_5-multires','N3908_CD4_BLUE_7-multires','O6218_CD3_YELLOW_7-multires','N3908_CD8_PURPLE_8-multires','O6218_CD4_BLUE_6-multires' ]
for slide in slides_list:
    folder_name1= '/scratch/KurcGroup/mazhao/tiles_slide/single_stain/'
    folder_name = folder_name1+slide+'/'
    lists=os.listdir(folder_name)
#('/home/lehhou/maozheng/data/test_set/TCGA_imgs_idx5_resized1.9076005961251865/')
    print(len(lists))
    f=open(slide+'.csv','w')
    f.write('patch_path,label'+'\n')
    for i in lists:
        #f.write('/home/lehhou/maozheng/data/test_set/selected_tiles_for_labeling/'+i+',1'+'\n')
        if i.endswith('png'):
            f.write(folder_name+i+',1'+'\n')
    f.close()
