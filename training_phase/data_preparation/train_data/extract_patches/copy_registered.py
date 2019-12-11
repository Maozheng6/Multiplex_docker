import os
import shutil
import glob
infolder='/scratch/KurcGroup/mazhao/ICCV/data_multiplex/dots_dan/60_registered_pairs/'
name_dict={'N22034':'N22034_90_Scale_bar_is_set_wrong','O3105':'O3105_10'}
files_to_copy=glob.glob(infolder+'*wsi.png')
counter=1
exist_count = 0
for file_i in files_to_copy:
    slide_name=os.path.basename(file_i).split('-')[0]
    if slide_name in name_dict.keys():
        slide_name = name_dict[slide_name]
    new_name = os.path.basename(file_i).split('multires_')[1]
    new_name = new_name.split('_wsi.png')[0]+'.png'
    print(counter,slide_name,new_name)
    if os.path.exists('./'+slide_name+'/'+new_name):
        print('./'+slide_name+'/'+new_name+' exists!!')
        exist_count+=1
        print('exist_count',exist_count)
    else:
        #shutil.copyfile(file_i,'./'+slide_name+'/'+new_name)
        pass
    counter+=1


