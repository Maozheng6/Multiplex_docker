import cv2
import numpy as np


def disRGBimg(img,c2):
    img=img[::-1,:,:].astype(np.float32)
    for i in range(3):
        img[i,:,:]=(img[i,:,:]-c2[i])*(img[i,:,:]-c2[i])
    #print('np.max(img),np.min(img)',np.max(img),np.min(img))
    return (255-np.sqrt(np.sum(img,axis=0)/(3*255*255))*255).astype('uint8')
    #return (1-np.sqrt(np.sum(img,axis=0)/(3*255*255))).astype('float')
    
img=cv2.imread('2893_cd20h_cd3h_cd8h.png')
print(img.shape)

def L2_5layer(img):
    #img is from following:
    #    naip_tile=cv2.imread(naip_fn)/255.0
    #    naip_tile=np.swapaxes(naip_tile,2,0)
    #    naip_tile=np.swapaxes(naip_tile,2,1)
    stain_OD=np.array([[82.60197,90.8649,98.27977],[14.87054,138.48598,65.88664],[ 17.725187,24.566036,76.55223],[116.36674,16.986359, 12.227117],[35.247486,82.52399,34.31515]])
    stain_RGB=np.exp(-stain_OD*np.log(255)/255)*255-1

    #img=np.swapaxes(img,0,2)
    #img=np.swapaxes(img,1,2)
    img=img*255
    
    L2=np.zeros((img.shape[1],img.shape[2],6))
    for i in range(len(stain_RGB)):
        
        L2_i=disRGBimg(img,stain_RGB[i,:])/255.0
        #print(img.shape,L2_i.shape,stain_RGB[i,:])
        L2[:,:,i]=L2_i
        #print(disRGBimg(img,stain_RGB[i]))
        #cv2.imwrite(str(i)+'_L2.png',disRGBimg(img,stain_RGB[i,:]))#L2_i*255).astype('uint8'))
    return L2

