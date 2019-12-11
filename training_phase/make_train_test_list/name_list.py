import os
file_path='/scratch/KurcGroup/mazhao/ICCV_NEW_DOTS_data_code_v2_regi-wsi/output_patches_60_wsi_0.84_max_patch_5/'
test_set='/scratch/KurcGroup/mazhao/ICCV/data_multiplex/dots_dan/test_set/'
if not os.path.exists(test_set):
    os.makedirs(test_set)
lists=os.listdir(file_path)
#('/mnt/blobfuse/cnn-minibatches/maozheng_data/maozheng_tumor_region_ratio_240')
print(len(lists))
f=open('maozheng_patch_list_multiplex_dots_nounknow_wsi_0.84_max_patch_5.txt','w')
f1=open('new_dots_test_has_unknown.csv','w')
f1.write('patch_path,label'+'\n')
n1=0
n2=0
for i in lists:
    if i.endswith('npy') and not i.startswith('Dots-O0135'):
        f.write(file_path+i+'\n')
        n1+=1
    if i.endswith('npy') and i.startswith('Dots-O0135'):
        f1.write(file_path+i+',1'+'\n')
        n2+=1
        os.system('cp '+file_path+i+' '+test_set+i)
    #('/mnt/blobfuse/cnn-minibatches/maozheng_data/maozheng_tumor_region_ratio_240/'+i+'\n')
print(n1,n2)
f.close()
f1.close()
