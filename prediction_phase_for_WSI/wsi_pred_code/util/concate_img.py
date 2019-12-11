import cv2
import numpy as np

im1 = cv2.imread('/scratch/KurcGroup/mazhao/tiles_slide/O0135-multires/88001_28001_4000_4000_0.25_1_SEG.png')
im1 = cv2.resize(im1,(2000,2000))
im2 = cv2.imread('/scratch/KurcGroup/mazhao/tiles_slide/O0135-multires/92001_28001_4000_4000_0.25_1_SEG.png')
im2 = cv2.resize(im2,(2000,2000))
conc = np.concatenate((im1,im2),1)
conc = conc[:,:,::-1]
conc = conc[:,1600:3600,:]
cv2.imwrite('22.png',conc)
