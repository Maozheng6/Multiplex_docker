import numpy as np
names=[x.rstrip('\n') for x in open('maozheng_tumor_patch_list.txt')]
count=0
for aname in names:
    count+=1
    print(count/len(names))
    a=np.load(aname)
    a=a.astype('float32')
    a[0,0:3,:,:]=a[0,0:3,:,:]/255
    np.save(aname,a)
    