import cv2
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
def convert_to_optical_densities(rgb,r0,g0,b0):
    OD = rgb.astype(float)
    OD[:,:,0] /= r0
    OD[:,:,1] /= g0
    OD[:,:,2] /= b0
    OD=np.clip(OD,0.001,1)
    
    return -np.log(OD)
def convert_to_optical_densities_1(rgb,r0,g0,b0):
    OD = rgb.astype(float)
    OD[:,:,0] = -255*np.log((OD[:,:,0]+1)/r0)/np.log(255)
    OD[:,:,1] = -255*np.log((OD[:,:,1]+1)/g0)/np.log(255)
    OD[:,:,2] = -255*np.log((OD[:,:,2]+1)/b0)/np.log(255)
    OD=np.clip(OD,0,255)
    
    return OD
def norm(matrix):
    for i in range(matrix.shape[0]):
        temp=matrix[i][:]
        matrix[i][:]=temp/np.sqrt(np.sum(temp*temp))
    return matrix
def color_deconvolution(rgb,stain_OD,r0,g0,b0,verbose=False):
    #stain_OD = np.asarray([[0.18,0.20,0.08],[0.01,0.13,0.0166],[0.10,0.21,0.29]]) #hematoxylin, eosyn, DAB
    #stain_OD=np.array([[116.36674,16.986359, 12.227117],[22.989534,19.66809,15.506229],[35.247486,82.52399,34.31515]])
    
    #n = []
    #for r in stain_OD:
        #n.append(r/norm(r))

    #normalized_OD = np.asarray(n)
    normalized_OD=norm(stain_OD)
    D = inv(normalized_OD)

    OD = convert_to_optical_densities_1(rgb,r0,g0,b0)

    ODmax = np.max(OD,axis=2)
    #plt.figure()
    #plt.imshow(ODmax>.1)

    # reshape image on row per pixel
    rOD = np.reshape(OD,(-1,3))
    # do the deconvolution
    rC = np.dot(rOD,D)
    #rescale
    rC=np.clip(rC,0,255)
    rC=np.exp(-((rC - 255.0) * np.log(255) / 255.0))
    rC=np.clip(rC,0,255).astype('uint8')
    #restore image shape
    C = np.reshape(rC,OD.shape)

    #remove problematic pixels from the the mask
    ODmax[np.isnan(C[:,:,0])] = 0
    ODmax[np.isnan(C[:,:,1])] = 0
    ODmax[np.isnan(C[:,:,2])] = 0
    ODmax[np.isinf(C[:,:,0])] = 0
    ODmax[np.isinf(C[:,:,1])] = 0
    ODmax[np.isinf(C[:,:,2])] = 0

    return (ODmax,C,normalized_OD)
    
def show_stain(normalized_OD,stain_intensity):
    for i in range(3):
        H=stain_intensity[:,:,i]
        print('normalized_OD[i][0]',normalized_OD[i][0])
        R=255-(255-H)*normalized_OD[i][0]
        print('R.shape',R.shape)
        print('H.shape',H.shape)
        G=255-(255-H)*normalized_OD[i][1]
        B=255-(255-H)*normalized_OD[i][2]
        img=np.zeros((H.shape[0],H.shape[1],3))
        img[:,:,0]=R
        img[:,:,1]=G
        img[:,:,2]=B
        print(img.shape)
        img=img[:,:,::-1]
        #img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('deconve_'+str(i)+'.png',img)
def save_stain_intensity(normalized_OD,stain_intensity):
    for i in range(3):
        H=255-stain_intensity[:,:,i]
        cv2.imwrite('./deconve_'+str(i)+'.png',H)



def get_deconve(img):
    img=img*255
    print('img.shape11',img.shape)
    img=np.swapaxes(img,2,0)
    img=np.swapaxes(img,1,0)
    img=img[:,:,::-1]
    print('img.shape22',img.shape)
    stain_OD_BCP=np.array([[82.60197,90.8649,98.27977],[116.36674,16.986359, 12.227117],[35.247486,82.52399,34.31515]])
    stain_OD_RYP=np.array([[14.87054,138.48598,65.88664],[ 17.725187,24.566036,76.55223],[35.247486,82.52399,34.31515]])
    (o,stain_intensity_RYP,normalized_OD)=color_deconvolution(img,stain_OD_RYP,255,255,255)
    (o,stain_intensity_BCP,normalized_OD)=color_deconvolution(img,stain_OD_BCP,255,255,255)
    Deconve=np.zeros((img.shape[0],img.shape[1],6))

    Deconve[:,:,0]=(255-stain_intensity_BCP[:,:,0])/255.0
    Deconve[:,:,1]=(255-stain_intensity_RYP[:,:,0])/255.0
    Deconve[:,:,2]=(255-stain_intensity_RYP[:,:,1])/255.0
    Deconve[:,:,3]=(255-stain_intensity_BCP[:,:,1])/255.0
    Deconve[:,:,4]=(255-stain_intensity_RYP[:,:,2])/255.0

    return Deconve

#img=cv2.imread('./851_cd16h_cd20h_cd3h_cd8h.png')
#img=img[:,:,::-1]
'''
for i in range(3):
    H=c[:,:,i]
    print(H)
    cv2.imwrite(str(i)+'.png',(H*255).astype('uint8'))
'''