import numpy as np
def color_augmentation(img):

        img=img.astype('float')
        beta=np.random.uniform(-0.1,0.1)
        alpha=np.random.uniform(-25.5,25.5)
        print('ba1',beta,alpha)
        img[:,:,0]=img[:,:,0]+beta*img[:,:,0]+alpha
        beta=np.random.uniform(-0.1,0.1)
        alpha=np.random.uniform(-25.5,25.5)
        print('ba2',beta,alpha)
        img[:,:,1]=img[:,:,1]+beta*img[:,:,1]+alpha
        beta=np.random.uniform(-0.1,0.1)
        alpha=np.random.uniform(-25.5,25.5)
        print('ba3',beta,alpha)
        img[:,:,2]=img[:,:,2]+beta*img[:,:,1]+alpha
        img=np.clip(img,0,255)
        img=img.astype('uint8')
        return img

import cv2
import sys
img=cv2.imread(sys.argv[1])
img=color_augmentation(img)
cv2.imwrite(sys.argv[2],img)


