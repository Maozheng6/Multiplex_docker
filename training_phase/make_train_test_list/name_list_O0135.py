import os
file_path='/scratch/KurcGroup/mazhao/ICCV/data_multiplex/dots_dan/big_path_with_dots_testset_O0135/'
lists=os.listdir(file_path)
#('/mnt/blobfuse/cnn-minibatches/maozheng_data/maozheng_tumor_region_ratio_240')
print(len(lists))
#f=open('maozheng_patch_list_multiplex_dots_has_unknow.txt','w')
f1=open('new_dots_test_O0135.csv','w')
f1.write('patch_path,label'+'\n')
n1=0
n2=0
for i in lists:

    if i.endswith('resized.png') :
        f1.write(file_path+i+',1'+'\n')
        n2+=1
        #os.system('cp '+file_path+i+' '+test_set+i)
    #('/mnt/blobfuse/cnn-minibatches/maozheng_data/maozheng_tumor_region_ratio_240/'+i+'\n')
print(n1,n2)
#f.close()
f1.close()
