import numpy as np
import os
lists=os.listdir('/mnt/blobfuse/train-output/ByMZ/multiplex10_patches_labelled/multiplex_training_with_background_label/')
#('/mnt/blobfuse/cnn-minibatches/maozheng_data/maozheng_tumor_region_ratio_240')
print(len(lists))
count=np.zeros((1,10))
for i in lists:
    a=i.rstrip('.npy')
    a=a.split('_')
    a=a[1:]
    a=[float(x) for x in a]
    a=np.array(a)
    count=count+a
print(count)
