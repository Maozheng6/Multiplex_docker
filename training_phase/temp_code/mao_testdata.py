import numpy as np
b=[x.rstrip('\n') for x in open('/mnt/blobfuse/cnn-minibatches/lym_minipatch_pred_v0_list.txt')]
print(len(b))
values=[]
for i,v in enumerate(b):
    if i%1000==0:
        temp=np.load(v)
        temp=temp[0,-1,120,120]
        values.append(temp)
print(values)
print(np.unique(values))
