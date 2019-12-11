import cv2
import numpy as np
from skimage.morphology import remove_small_holes
imgname='999_0.0_1.0_0.0_1.0_1.0_0.0_0.0_1.0_1.0_0.0.npyforeground_dis_bi2.png'
img=cv2.imread(imgname)
kernel=np.ones((3,3))
img_di=cv2.erode(img,kernel,iterations=1)
cv2.imwrite('img_di.png',img_di)
img_filled=remove_small_holes((img_di>0))
cv2.imwrite('img_filled.png',img_filled*255)
print(np.max(img_filled))
img_ero=cv2.erode(img_filled.astype('uint8')*255,kernel,iterations=2)
cv2.imwrite('img_ero.png',img_ero.astype('uint8'))
img_ero=cv2.erode(img_ero.astype('uint8')*255,kernel,iterations=1)
cv2.imwrite('img_ero1.png',img_ero.astype('uint8')*255)