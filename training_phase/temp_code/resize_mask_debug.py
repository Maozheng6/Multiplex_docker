import numpy as np
import cv2
mask = np.zeros((10,10))
mask[2,3]=1
mask[8,9]=1
def resize_mask(mask,times):
	idx=np.nonzero(mask)
	print(idx)
	idx2=[x/times for x in idx]
	idx=idx2
	print(idx)
	print(list(range(len(idx[0]))))
	mask2=np.zeros((int(mask.shape[0]/times),int(mask.shape[1]/times)))
	for i in range(len(idx[0])):
		row=int(idx[0][i])
		col=int(idx[1][i])
		mask2[row,col]=1

	return mask2
def to_highres_mask(im, class_num,block_size_m,unet_level)
    one_hot = np.zeros((class_num+1, im.shape[0],im.shape[0]), dtype=np.float32)
    #print('one hot',one_hot.shape,im.shape,class_num)
    for class_id in range(class_num+1):
        one_hot[class_id, :, :] = (im == class_id+1).astype(np.float32)
    print(one_hot)
    resized_one_hot = np.zeros((class_num+1, block_size_m, block_size_m), dtype=np.float32)
    for class_id in range(class_num+1):
        resized_one_hot[class_id, :, :] = cv2.resize(one_hot[class_id, :, :],(block_size_m,block_size_m),cv2.INTER_LINEAR)
    resized_one_hot=(resized_one_hot>0.5).astype('int')
    #resized_one_hot=np.swapaxes(resized_one_hot,0,2)
    #resized_one_hot=np.swapaxes(resized_one_hot,0,1)
    #cv2.resize(resized_one_hot
    return resized_one_hot
'''
print(mask)
idx=np.nonzero(mask)
print(len(idx[0]))
print(list( range(len(idx[0]))))
print(mask[idx])
mask2=resize_mask(mask,2)
print(mask2)
'''
import numpy as np
im=np.random.rand(10,10)
im=(im>0.5).astype(int)
im2=np.random.rand(10,10)
im2=(im2>0.5).astype(int)*2
im[np.nonzero(im2)]=2
print(im)

re=to_highres_mask(im, 2,5,2)
print(re)
