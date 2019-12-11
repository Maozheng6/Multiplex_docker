import cv2
import numpy as np
import sys
print(sys.argv[1],sys.argv[2])
wsi=cv2.imread(sys.argv[1])
dan=cv2.imread(sys.argv[2])

def mu_sigma(im1):
    im1=np.resize(im1,(im1.shape[0]*im1.shape[1],im1.shape[2]))
    mu=np.mean(im1,0)
    sigma=np.std(im1,0)
    return mu,sigma

def diff_stat(wsi_stat,dan_stat):
    diff1_stat=np.array(wsi_stat)-np.array(dan_stat)
    diff2_stat=np.array(wsi_stat)/np.array(dan_stat)
    diff3_stat=diff2_stat
    diff3_stat[0,:]=diff1_stat[0,:]
    return diff3_stat

def mapping_dan2wsi(dan,wsi):
    dan_stat=mu_sigma(dan)
    wsi_stat=mu_sigma(wsi)
    print('dan_stat',dan_stat)
    print('wsi_stat',wsi_stat)


    diff3_stat = diff_stat(wsi_stat,dan_stat)
    print('diff3_stat',diff3_stat)
    print('para',diff3_stat[1,0],diff3_stat[0,0])
    dan[:,:,2]=dan[:,:,2]*diff3_stat[1,2]
    dan[:,:,1]=dan[:,:,1]*diff3_stat[1,1]
    dan[:,:,0]=dan[:,:,0]*diff3_stat[1,0]
    dan_stat_new=mu_sigma(dan)

    print('dan_stat_new',dan_stat_new)
    diff3_stat = diff_stat(wsi_stat,dan_stat_new)
    print('diff3_stat',diff3_stat)

    dan[:,:,2]=dan[:,:,2]+diff3_stat[0,2]
    dan[:,:,1]=dan[:,:,1]+diff3_stat[0,1]
    dan[:,:,0]=dan[:,:,0]+diff3_stat[0,0]
    dan_stat_new =mu_sigma(dan)
    diff3_stat = diff_stat(wsi_stat,dan_stat_new)
    print('diff3_stat',diff3_stat)
    return dan

dan=mapping_dan2wsi(dan,wsi)
cv2.imwrite('mapped_dan.png',dan)

