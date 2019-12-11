import cv2
import numpy as np
import os
import colorsys
import copy
def disRGBimg(img,c2):
    img=img[::-1,:,:].astype(np.float32)
    for i in range(3):
        img[i,:,:]=(img[i,:,:]-c2[i])*(img[i,:,:]-c2[i])
    #print('np.max(img),np.min(img)',np.max(img),np.min(img))
    return (255-np.sqrt(np.sum(img,axis=0)/(3*255*255))*255).astype('uint8')
def dis_stack_img(img,c2):
    #img=img[::-1,:,:].astype(np.float32)
    for i in range(6):
        img[i,:,:]=(img[i,:,:]-c2[i])*(img[i,:,:]-c2[i])
    #print('np.max(img),np.min(img)',np.max(img),np.min(img))
    return (255-np.sqrt(np.sum(img,axis=0)/(3*255*255))*255).astype('uint8')
 
def pencentile_and_pool(i, dis, percent,prefix,save_visual,naip_slice,patch_file,fore_ground,back_ground):
        visual_folder='/mnt/blobfuse/train-output/ByMZ/high_res_20back'
        color_iter_distance=[59.38414365, 47.0623548, 68.98451365, 61.70322829, 47.0623548 ]
        if not os.path.exists(visual_folder):
            os.makedirs(visual_folder)
        p=np.percentile(dis,percent)
        if prefix=='background':
            p=np.max((p,255-30))
        else:
            p=np.max((p,255-color_iter_distance[int(i/2)])) #35=np.sqrt(20^2+20^2+20^2)
        print('i',i,'p',p)
        s,dis_bi=cv2.threshold(dis,p,255,cv2.THRESH_BINARY)
        kernel=np.ones((3,3))
        dis_ero=cv2.dilate(dis_bi,kernel,iterations=1)
        if prefix=='background':
            dis_ero=cv2.erode(dis_ero,kernel,iterations=9)
        else:
            dis_ero=cv2.erode(dis_ero,kernel,iterations=2)
        dis_ero=(dis_ero>0).astype('uint8')
        
     
        pooled=self.maxpool(dis_ero,self.unet_level)
        pooled=pooled+back_ground
        pooled_bi=(pooled>0).astype('uint8')
        pooled_bi=pooled_bi-fore_ground
        pooled_bi=(pooled_bi>0).astype('uint8')
        
        
        pooled_bi_expand=self.expand_output_mao(pooled_bi*255,self.unet_level)
        naip_slice1=np.swapaxes(naip_slice,0,2)
        naip_slice1=np.swapaxes(naip_slice1,0,1)
                
        if self.save_visual==1:
            cv2.imwrite(os.path.join(visual_folder,os.path.basename(patch_file)+prefix+'_'+'dis_bi'+str(int(i/2))+'.png'),dis_bi)
            cv2.imwrite(os.path.join(visual_folder,os.path.basename(patch_file)+prefix+'_'+'dis_ero'+str(int(i/2))+'.png'),dis_ero*255)
            cv2.imwrite(os.path.join(visual_folder,os.path.basename(patch_file)+prefix+'_'+'pooled_bi'+str(int(i/2))+'.png'),pooled_bi_expand)

            if not os.path.exists(os.path.join(visual_folder,os.path.basename(patch_file)+prefix+'_'+'ori.png')):
                cv2.imwrite(os.path.join(visual_folder,os.path.basename(patch_file)+prefix+'_'+'ori.png'),(naip_slice1*255).astype('uint8'))
        return pooled_bi
        
#####color extraction
stain_OD=np.array([[82.60197,90.8649,98.27977],[14.87054,138.48598,65.88664],[ 17.725187,24.566036,76.55223],[116.36674,16.986359, 12.227117],[35.247486,82.52399,34.31515]])
stain_RGB=np.exp(-stain_OD*np.log(255)/255)*255-1
IHC_OD=np.array([26.784452,20.88244,14.707071])
IHC_RGB=np.exp(-IHC_OD*np.log(255)/255)*255-1


save_folder='./color_L2/'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
color_name=['CD16 dark brown','CD20 pink','CD3 yellow', 'CD4 cyan','CD8 purple']
inputfolder='./test_color_L2/'
imname='1475_1.0_0.0_0.0_1.0_1.0_0.0_1.0_0.0_1.0_0.0.npyforeground_ori.png'
#

for i in range(5):
   
       
        #print('img.shape',img.shape)
        #print('stain_RGB[i/2,:]',stain_RGB[int(i/2),:])
        img=cv2.imread(os.path.join(inputfolder,imname))
        #img=np.transpose(img)
        #
        print(img.shape)
        #dis=colorsys.rgb_to_hsv(img[2,:,:]/255.0,img[1,:,:]/255.0,img[0,:,:]/255.0)
        img[0,0,2]=stain_RGB[i,0]
        img[0,0,1]=stain_RGB[i,1]
        img[0,0,0]=stain_RGB[i,2]
        dis = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        dis=np.transpose(dis)
        hsv=copy.deepcopy(dis)
        dis=disRGBimg(dis,[dis[2,0,0],dis[1,0,0],dis[0,0,0]])
        dis=np.transpose(dis)
      
        img=np.transpose(img)
        rgb=copy.deepcopy(img)
        disrgb=disRGBimg(img,stain_RGB[i])
        disrgb=np.transpose(disrgb)
        
        #dis_IHC=disRGBimg(img,IHC_RGB)
        
        stack=np.zeros((6,rgb.shape[1],rgb.shape[2]))
        stack[0:3,:,:]=rgb[::-1,:,:]
        stack[3:6,:,:]=hsv[::-1,:,:]
        color_stack=np.zeros((6))
        color_stack[0:3]=stain_RGB[i]
        color_stack[3:6]=np.array([hsv[2,0,0],hsv[1,0,0],hsv[0,0,0]])
        dis_stack=dis_stack_img(stack,color_stack)
        dis_stack=np.transpose(dis_stack)
        print(np.max(dis),np.min(dis))
        cv2.imwrite(os.path.join(save_folder,imname[0:-4]+color_name[i]+'_hsv.png'),dis.astype('uint8'))
        cv2.imwrite(os.path.join(save_folder,imname[0:-4]+color_name[i]+'_rgb.png'),disrgb.astype('uint8'))
        cv2.imwrite(os.path.join(save_folder,imname[0:-4]+color_name[i]+'_stack.png'),dis_stack.astype('uint8'))
        