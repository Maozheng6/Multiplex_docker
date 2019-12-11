import sys
import os
import time
import cv2
import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

import cntk
import cntk.io
import string

import random
import numpy as np
import shapely
import shapely.geometry
import rasterio
import rasterio.mask
import glob
import scipy.ndimage as nd
from skimage.color import rgb2hed
from shapely.geometry import mapping
#from osgeo import gdal
from sklearn.utils import shuffle
from scipy import misc
from color_trans import dan_2_wsi

from DataHandle import get_nlcd_stats
cid, nlcd_dist, nlcd_var = get_nlcd_stats()
import random

def color_aug(color):
    n_ch = color.shape[0]
    contra_adj = 0.05
    bright_adj = 0.05

    ch_mean = np.mean(color, axis=(-1,-2), keepdims=True).astype(np.float32)

    contra_mul = np.random.uniform(1-contra_adj, 1+contra_adj, (n_ch,1,1)).astype(np.float32)
    bright_mul = np.random.uniform(1-bright_adj, 1+bright_adj, (n_ch,1,1)).astype(np.float32)

    color = (color - ch_mean) * contra_mul + ch_mean * bright_mul
    return color


#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
# Custom CNTK datasources
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------

class MyDataSource(cntk.io.UserMinibatchSource):
    ''' A minibatch source for loading pre-extracted batches '''
    def __init__(self, f_dim, l_dim, m_dim, c_dim, highres_only, edge_sigma, edge_loss_boost, patch_list, patch_list2_dots_only,unet_level=16,stain_num=1,start_stain=0,save_visual=0,channel_times=6,mu_times=1.0,sigma_times=1.0):
        # Record passed parameters for use in later methods
        self.f_dim, self.l_dim, self.m_dim, self.c_dim = f_dim, l_dim, m_dim, c_dim
        self.highres_only = highres_only
        # ...and a few more that we can infer from the values that were passed
        self.num_color_channels, self.block_size, _ = self.f_dim
        _,self.block_size_m,_=self.l_dim
        self.lwm_dim = (1, self.l_dim[1], self.l_dim[2])
        self.edge_sigma = edge_sigma
        self.edge_loss_boost = edge_loss_boost
        self.num_nlcd_classes, self.num_landcover_classes = self.c_dim
        self.stain_num=stain_num
        self.start_stain=start_stain
        self.unet_level=unet_level
        self.save_visual=save_visual
        self.channel_times=channel_times
        self.mu_times=mu_times
        self.sigma_times=sigma_times

        self.lc_6_layer, _, _=self.l_dim


        # Record the stream information.
        self.fsi = cntk.io.StreamInformation(
            'features', 0, 'dense', np.float32, (self.num_color_channels*self.channel_times, self.block_size, self.block_size))
        self.lsi = cntk.io.StreamInformation(
            'landcover', 0, 'dense', np.float32, ((self.stain_num+1)*self.channel_times,self.block_size_m, self.block_size_m))
        self.lwi = cntk.io.StreamInformation(
            'lc_weight_map', 0, 'dense', np.float32, self.lwm_dim)
        self.msi = cntk.io.StreamInformation(
            'masks', 1, 'dense', np.float32, (5, self.block_size_m*self.channel_times, self.block_size_m))
        self.csi = cntk.io.StreamInformation(
            'interval_centers', 1, 'dense', np.float32, (5*self.channel_times,))
        self.rsi = cntk.io.StreamInformation(
            'interval_radii', 1, 'dense', np.float32, (5*self.channel_times,))

        self.patches1 = [line.rstrip('\n') for line in open(patch_list)]
        if patch_list2_dots_only == None:
            self.patches2 = []
        else:
            print(patch_list2_dots_only)
            self.patches2 = [line.rstrip('\n') for line in open(patch_list2_dots_only)]
        print('len(self.patches1)',len(self.patches1))
        print('len(self.patches2)',len(self.patches2))
        self.dots_patch_ratio = float(len(self.patches2))/float(len(self.patches2)+len(self.patches1))
        print('self.dots_patch_ratio',self.dots_patch_ratio)
        self.all_patches =[]
        self.all_patches.append(self.patches1)
        self.all_patches.append(self.patches2)
        print('3 len:',len(self.all_patches),len(self.all_patches[0]),len(self.all_patches[1]))

        if self.highres_only:
            print('high-res only. To-Do')
        assert(len(self.all_patches) > 0)
        self.patch_list_for_all_classes = self.split_patch_per_class(self.all_patches)
        self.freq_control_arr = np.zeros((self.num_nlcd_classes*self.stain_num,), dtype=np.float32)
        self.class_count_arr = [np.zeros((5,self.num_landcover_classes), dtype=np.float32)]
        self.batch_count = 0
        self.class_iter_idx = 0

        super(MyDataSource, self).__init__()

    #def split_patch_per_class_ori(self, all_patches):
    #    patch_list_for_all_classes = []
    #    for nlcd_class in range(self.num_nlcd_classes):
    #        patch_list_per_class = [x for x in all_patches if \
    #                    int(os.path.basename(x).split('.npy')[0].split('_')[0])==nlcd_class]
    #        print('List len {} for class {}'.format(len(patch_list_per_class), nlcd_class))
    #        patch_list_for_all_classes.append(patch_list_per_class)
    #    return patch_list_for_all_classes
    def split_patch_per_class(self, all_patches):
        patch_list_for_all_classes = []

        for nlcd_class in range(10):
            print('len( all_patches[0])',len( all_patches[0]))
            patch_list_per_class = [x for x in all_patches[0] if \
                        os.path.basename(x).split('.npy')[0].split('_')[nlcd_class+1]=='1.0']
            print('List len {} for class {}'.format(len(patch_list_per_class), nlcd_class))
            patch_list_for_all_classes.append(patch_list_per_class)
        patch_list_for_all_classes.append(all_patches[1])
        return patch_list_for_all_classes

    def stream_infos(self):
        ''' Define the streams that will be returned by the minibatch source '''
        return [self.fsi, self.lsi, self.lwi, self.msi, self.csi, self.rsi]

    def to_one_hot(self, im, class_num):
        one_hot = np.zeros((class_num, im.shape[-2], im.shape[-1]), dtype=np.float32)
        for class_id in range(class_num):
            one_hot[class_id, :, :] = (im == class_id).astype(np.float32)
        return one_hot

    def maxpool(self,dis_masked,times):
        new=np.zeros((int(dis_masked.shape[0]/times),int(dis_masked.shape[1]/times)))
        for i in range(0,dis_masked.shape[0],times):
            for j in range(0,dis_masked.shape[1],times):
                new[int(i/times),int(j/times)]=np.max(dis_masked[i:i+times,j:j+times])
        return new

    def count_pool(self,dis_masked,times):
        ratio=0.8
        new=np.zeros((int(dis_masked.shape[0]/times),int(dis_masked.shape[1]/times))).astype('uint8')
        #new_count=np.zeros((int(dis_masked.shape[0]/times),int(dis_masked.shape[1]/times)))
        for i in range(0,dis_masked.shape[0],times):
            for j in range(0,dis_masked.shape[1],times):
                if np.sum(dis_masked[i:i+times,j:j+times])>ratio*times*times:
                    new[int(i/times),int(j/times)]=1
                #new_count[int(i/times),int(j/times)]=np.sum(np.sum(dis_masked[i:i+times,j:j+times]))/float(times*times)
        return new#,new_count

    def disRGBimg(self,img,c2):
        img=img[::-1,:,:].astype(np.float32)*255
        for i in range(3):
            img[i,:,:]=(img[i,:,:]-c2[i])*(img[i,:,:]-c2[i])
        #print('np.max(img),np.min(img)',np.max(img),np.min(img))
        return (255-np.sqrt(np.sum(img,axis=0)/(3*255*255))*255).astype('uint8')
    def expand_output_mao(self,output,times):
        row=output.shape[0]
        col=output.shape[1]
        #print(output.shape)
        new_output=np.ones((output.shape[0]*times,output.shape[1]*times))
        for i in range(row):
            for j in range(col):
                new_output[i*times:(i+1)*times,j*times:(j+1)*times]=output[i,j]
        return new_output
    def pencentile_and_pool(self,i, dis, percent,prefix,save_visual,naip_slice,patch_file,fore_ground,back_ground,erode_iter):
        visual_folder='../../DOTS_output/visual'
        color_iter_distance=[36.03450664, 47.05507921, 35.75816175, 36.03450664, 41.18899157]
        #, 35.75816175]#[59.38414365, 47.0623548, 68.98451365, 61.70322829, 47.0623548 ]
        if not os.path.exists(visual_folder):
            os.makedirs(visual_folder)
        p=np.percentile(dis,percent)
        if prefix=='background':
            p=np.max((p,255-30))
        else:
            p=np.max((p,255-color_iter_distance[int(i/2)])) #35=np.sqrt(20^2+20^2+20^2)

        s,dis_bi=cv2.threshold(dis,p,255,cv2.THRESH_BINARY)
        kernel=np.ones((3,3))
        dis_ero=cv2.dilate(dis_bi,kernel,iterations=1)
        if prefix=='background':
            dis_ero=cv2.erode(dis_ero,kernel,iterations=3)
        else:
            dis_ero=cv2.erode(dis_ero,kernel,iterations=erode_iter+1)
        dis_ero=(dis_ero>0).astype('uint8')

        if prefix!='background':
            pooled=self.maxpool(dis_ero,self.unet_level)
        else:
            pooled=self.count_pool(dis_ero,self.unet_level,0.9)
        pooled=pooled+back_ground
        pooled_bi=(pooled>0).astype('uint8')
        pooled_bi=pooled_bi-fore_ground
        pooled_bi=(pooled_bi>0).astype('uint8')


        pooled_bi_expand=self.expand_output_mao(pooled_bi*255,self.unet_level)
        naip_slice1=np.swapaxes(naip_slice,0,2)
        naip_slice1=np.swapaxes(naip_slice1,0,1)

        if self.save_visual==1:

            if prefix!='background':
                cv2.imwrite(os.path.join(visual_folder,os.path.basename(patch_file)+prefix+'_'+'dis_bi'+str(int(i/2))+'.png'),dis_bi)
                cv2.imwrite(os.path.join(visual_folder,os.path.basename(patch_file)+prefix+'_'+'dis_ero'+str(int(i/2))+'.png'),dis_ero*255)
                cv2.imwrite(os.path.join(visual_folder,os.path.basename(patch_file)+prefix+'_'+'pooled_bi'+str(int(i/2))+'.png'),pooled_bi_expand)
                if not os.path.exists(os.path.join(visual_folder,os.path.basename(patch_file)+prefix+'_'+'ori.png')):
                    cv2.imwrite(os.path.join(visual_folder,os.path.basename(patch_file)+prefix+'_'+'ori.png'),(naip_slice1*255).astype('uint8'))
            else:
                cv2.imwrite(os.path.join(visual_folder,os.path.basename(patch_file)+prefix+'_'+'dis_bi_back'+str(int(i/2))+'.png'),dis_bi)
                cv2.imwrite(os.path.join(visual_folder,os.path.basename(patch_file)+prefix+'_'+'dis_ero_back'+str(int(i/2))+'.png'),dis_ero*255)
                cv2.imwrite(os.path.join(visual_folder,os.path.basename(patch_file)+prefix+'_'+'pooled_bi_back'+str(int(i/2))+'.png'),pooled_bi_expand)

        return pooled_bi
    def data_augmentation(self,img,label):
        if np.random.choice([0,1])>0.5:
            #vertical flip
            img = img[:,::-1,:]
            label = label[:,::-1,:]
        if np.random.choice([0,1])>0.5:
            #horizontal flip
            img = img[:,:,::-1]
            label = label[:,:,::-1]
        if np.random.choice([0,1])>0.5:
            img = np.rot90(img,1,(1,2))
            label = np.rot90(label,1,(1,2))
        if np.random.choice([0,1])>0.5:
            img = np.rot90(img,1,(2,1))
            label = np.rot90(label,1,(2,1))
        return img, label

    def color_augmentation(self,img):
        img=img.astype('float')
        beta=np.random.uniform(-0.2,0.2)
        alpha=np.random.uniform(-0.1,0.1)
        #print('ba1',beta,alpha)
        img[:,:,0]=img[:,:,0]+beta*img[:,:,0]+alpha
        beta=np.random.uniform(-0.2,0.2)
        alpha=np.random.uniform(-0.1,0.1)
        #print('ba2',beta,alpha)
        img[:,:,1]=img[:,:,1]+beta*img[:,:,1]+alpha
        beta=np.random.uniform(-0.2,0.2)
        alpha=np.random.uniform(-0.1,0.1)
        #print('ba3',beta,alpha)
        img[:,:,2]=img[:,:,2]+beta*img[:,:,1]+alpha
        img=np.clip(img,0,1.0)
        img=img.astype('float')
        return img

    def sample_slices_from_list(self, patch_list,block_size_m):

        try:
            patch_file = random.sample(patch_list, 1)[0]
            minipatch = np.load(patch_file).astype(np.float32)
        except:
            patch_file = random.sample(patch_list, 1)[0]
            minipatch = np.load(patch_file).astype(np.float32)
        while np.isnan(minipatch).any() or np.isinf(minipatch).any():
            logging.warning("Loaded one patch with nan or inf {}".format(patch_file))
            patch_file = random.sample(patch_list, 1)[0]
            minipatch = np.load(patch_file).astype(np.float32)
        #resize
        #minipatch=np.resize(minipatch,(1,4,160,160))
        #minipatch=minipatch.astype('int')
        naip_slice = minipatch[0, 0:3, ...]
        #nlcd_slice = np.squeeze(minipatch[0, -1, ...])
        #mao:for no uppooling

        nlcd_slice = np.squeeze(minipatch[0, 3:13, 0:block_size_m,0:block_size_m])
        #nlcd_slice = np.ones((5, block_size_m,block_size_m))
        ##########multi
        '''
        nlcd_slice_new=np.zeros((5,block_size_m,block_size_m))
        for i in range(5):
            if nlcd_slice[i*2,0,0]==0:
                nlcd_slice_new[i,:,:]=0
            else:
                nlcd_slice_new[i,:,:]=1
        nlcd_slice=nlcd_slice_new
        '''
        ############


        lc_slice = np.zeros((self.stain_num+1,block_size_m,block_size_m), dtype=np.float32)
        lc_background=np.squeeze(minipatch[0, -1,:,:])
        lc_background=self.count_pool(lc_background,self.unet_level )

        #####color extraction
        #True => F1=0.629
        #False => F1=0.613
        color_extraction = True
        if color_extraction==True:
            stain_OD=np.array([[82.60197,90.8649,98.27977],[14.87054,138.48598,65.88664],[ 17.725187,24.566036,76.55223],[116.36674,16.986359, 12.227117],[35.247486,82.52399,34.31515]])
            stain_RGB=np.exp(-stain_OD*np.log(255)/255)*255-1
            IHC_OD=np.array([26.784452,20.88244,14.707071])
            IHC_RGB=np.exp(-IHC_OD*np.log(255)/255)*255-1
            foreground=np.zeros((block_size_m,block_size_m))

            for i in range(2*self.start_stain,2*self.start_stain+5*2,2):
                if os.path.basename(patch_file).split('.npy')[0].split('_')[i+1]=='1.0':
                    img=naip_slice
                    #print('img.shape',img.shape)
                    #print('stain_RGB[i/2,:]',stain_RGB[int(i/2),:])
                    dis=self.disRGBimg(img,stain_RGB[int(i/2),:])
                    dis_IHC=self.disRGBimg(img,IHC_RGB)
                    if  i==2:
                        percent=99.75
                        erode_iter=1
                    if i==6 :
                        percent=99.5
                        erode_iter=1
                    if i==0:
                        percent=99.5
                        erode_iter=1
                    if i==4:
                        percent=99.75
                        erode_iter=1
                    if i==8:
                        percent=99
                        erode_iter=2

                    percentb=80
                    save_visual=0


                    fore_gd=self.pencentile_and_pool(i, dis, percent, 'foreground',save_visual,naip_slice,patch_file,0,0,erode_iter)
                    lc_slice[int((i-2*self.start_stain)/2),:,:]=(fore_gd>0).astype(np.float32)

            lc_slice[-1,:,:]=(lc_background>0).astype(np.float32)


            order_color=[3,2,1,4,0,5]
            for i in range(len(order_color)-1):
                for j in range(i+1,len(order_color)):
                    lc_slice[order_color[j],:,:]=lc_slice[order_color[j],:,:]-lc_slice[order_color[i],:,:]


            lc_slice=np.clip(lc_slice,0.0,1.0)

            #nlcd_class_count:list of len 5, each element is a vector of 'num_nlcd_classes' many elements

        #note: the last 4 channels are fake labels
        nlcd_slice=np.concatenate((nlcd_slice,nlcd_slice[0:4,:,:]),0)
        #print('nlcd_slice.shape',nlcd_slice.shape)


        nlcd_class_count=[]
        for i in range(5):
            nlcd_class_count.append(np.zeros((self.num_landcover_classes,), dtype=np.float32))
        for i in range(5):
            for class_id in range(self.num_landcover_classes):
                nlcd_class_count[i][class_id] = np.sum(nlcd_slice[2*i+class_id,:,:]==1)
        naip_slice,lc_slice=self.data_augmentation(naip_slice,lc_slice)
        #naip_slice = self.color_augmentation(naip_slice)
        return naip_slice, nlcd_slice, lc_slice, nlcd_class_count


    def sample_slices_from_list_dotsimg(self, patch_list,block_size_m):
        def resize_mask(mask,times):
            idx=np.nonzero(mask)
            #print(idx)
            idx2=[x/times for x in idx]
            idx=idx2
            #print(idx)
            #print(list(range(len(idx[0]))))
            mask2=np.zeros((int(mask.shape[0]/times),int(mask.shape[1]/times)))
            for i in range(len(idx[0])):
                row=int(idx[0][i])
                col=int(idx[1][i])
                mask2[row,col]=1
            return mask2

        def to_highres_mask(im, class_num,block_size_m,unet_level):
            #print('one hot',one_hot.shape,im.shape,class_num)
            one_hot = np.zeros((class_num+1, im.shape[0],im.shape[0]))
            for class_id in range(class_num+1):
                one_hot[class_id, :, :] = (im == class_id+1).astype('float')
            resized_one_hot = np.zeros((class_num+1, block_size_m,block_size_m))
            for class_id in range(class_num+1):
                resized_one_hot[class_id, :, :] = cv2.resize(one_hot[class_id, :, :],(block_size_m,block_size_m),cv2.INTER_LINEAR)
            resized_one_hot=(resized_one_hot>0.5).astype('int')

            return resized_one_hot
        try:
            patch_file = random.sample(patch_list, 1)[0]
            print('patch_file',patch_file)
            minipatch = np.load(patch_file).astype(np.float32)
        except:
            patch_file = random.sample(patch_list, 1)[0]
            minipatch = np.load(patch_file).astype(np.float32)
        while np.isnan(minipatch).any() or np.isinf(minipatch).any():
            logging.warning("Loaded one patch with nan or inf {}".format(patch_file))
            patch_file = random.sample(patch_list, 1)[0]
            minipatch = np.load(patch_file).astype(np.float32)
        #resize
        minipatch=np.swapaxes(minipatch,2,0)
        minipatch=np.swapaxes(minipatch,1,2)
        naip_slice = minipatch[ 0:3, ...].astype('float')
        #to bgr
        naip_slice = naip_slice[::-1,:,:]
        #is wsi?
        #print(' os.path.basename(os.path.dirname(patch_file))', os.path.basename(os.path.dirname(patch_file)))
        if os.path.basename(os.path.dirname(patch_file))== 'output_patches_nounknown_registered-wsi_6-7ratio_60wsi' or os.path.basename(os.path.dirname(patch_file))=='output_patches_nounknown_registered-wsi_6-7ratio' or os.path.basename(os.path.dirname(patch_file))== 'output_patches_nounknown_registered-wsi_6-7ratio_10wsi':

            is_wsi = 1
        else:
            is_wsi = 0
        #print('is_wsi',is_wsi,os.path.basename(os.path.dirname(patch_file)))

        #color transfer
        if 0:# is_wsi==0 and np.random.choice([0,1])>0.5 :
            domain_shift = 1

            naip_slice = np.swapaxes(naip_slice,0,2)
            naip_slice = np.swapaxes(naip_slice,0,1)

            naip_slice = dan_2_wsi(naip_slice)

            naip_slice = np.swapaxes(naip_slice,0,2)
            naip_slice = np.swapaxes(naip_slice,1,2)

        else:
            domain_shift = 0

        naip_slice = naip_slice/255.0
        #nlcd_slice = np.squeeze(minipatch[0, -1, ...])
        #mao:for no uppooling
        nlcd_slice=np.zeros((2*self.stain_num,block_size_m,block_size_m))
        for nn in range(self.stain_num):
                nlcd_slice[nn*2]+=1

        lc_slice_dots = np.squeeze(minipatch[ 3,:,:])
        lc_slice_dots = to_highres_mask(lc_slice_dots,self.stain_num,block_size_m,self.unet_level)


        #lc_background=self.count_pool(lc_background,self.unet_level )
        '''
        lc_slice = np.zeros((self.stain_num+1,block_size_m,block_size_m), dtype=np.float32)
        #####color extraction
        stain_OD=np.array([[82.60197,90.8649,98.27977],[14.87054,138.48598,65.88664],[ 17.725187,24.566036,76.55223],[116.36674,16.986359, 12.227117],[35.247486,82.52399,34.31515]])
        stain_RGB=np.exp(-stain_OD*np.log(255)/255)*255-1
        IHC_OD=np.array([26.784452,20.88244,14.707071])
        IHC_RGB=np.exp(-IHC_OD*np.log(255)/255)*255-1
        foreground=np.zeros((block_size_m,block_size_m))

        for i in range(2*self.start_stain,2*self.start_stain+(self.stain_num-2)*2,2):
            if int(os.path.basename(patch_file).split('.npy')[0].split('_')[i+1])>=5:
                img=naip_slice
                #print('img.shape',img.shape)
                #print('stain_RGB[i/2,:]',stain_RGB[int(i/2),:])
                dis=self.disRGBimg(img,stain_RGB[int(i/2),:])
                dis_IHC=self.disRGBimg(img,IHC_RGB)
                if  i==2:
                    percent=99.75
                    erode_iter=0
                if i==6 :
                    percent=99.5
                    erode_iter=1
                if i==0:
                    percent=99.5
                    erode_iter=1
                if i==4:
                    percent=99.5
                    erode_iter=1
                if i==8:
                    percent=99
                    erode_iter=2
                if i==10:
                    percent=99.5
                    erode_iter=1
                percentb=80
                save_visual=0


                fore_gd=self.pencentile_and_pool(i, dis, percent, 'foreground',save_visual,naip_slice,patch_file,0,0,erode_iter)
                lc_slice[int((i-2*self.start_stain)/2),:,:]=(fore_gd>0).astype(np.float32)

        lc_slice[-1,:,:]=(lc_background>0).astype(np.float32)


        order_color=[3,2,1,4,0,5]
        for i in range(len(order_color)-1):
            for j in range(i+1,len(order_color)):
                lc_slice[order_color[j],:,:]=lc_slice[order_color[j],:,:]-lc_slice[order_color[i],:,:]

        lc_slice+=lc_slice_dots
        lc_slice=np.clip(lc_slice,0.0,1.0)
        '''
        lc_slice=lc_slice_dots
        #save_temp='../../output/lc_slice/'
        #if not os.path.exists(save_temp):
        #    os.makedirs(save_temp)
        #np.save(save_temp + os.path.basename(patch_file)[0:-4]+'_1.npy',lc_slice)
        #np.save(save_temp + os.path.basename(patch_file),minipatch)
        nlcd_class_count=[]
        for i in range(5):
            nlcd_class_count.append(np.zeros((self.num_landcover_classes,), dtype=np.float32))
        for i in range(5):
            for class_id in range(self.num_landcover_classes):
                nlcd_class_count[i][class_id] = np.sum(nlcd_slice[2*i+class_id,:,:]==1)

        #nlcd_class_count:list of len 5, each element is a vector of 'num_nlcd_classes' many elements
        naip_slice,lc_slice=self.data_augmentation(naip_slice,lc_slice)

        #naip_slice = self.color_augmentation(naip_slice)
        return naip_slice, nlcd_slice, lc_slice, nlcd_class_count


    def print_class_count(self):
        self.batch_count += 1
        if self.batch_count % 200 == 0:
            class_percent = (0.5+self.class_count_arr/np.sum(self.class_count_arr) \
                    * 100).astype(np.uint8)
            logging.info("NLCD class counts: {}".format(class_percent))

    def get_random_instance(self):
        naip_slice, nlcd_slice, lc_slice, nlcd_class_count = \
            self.sample_slices_from_list(self.patch_list_for_all_classes[
                np.random.randint(0, len(self.patch_list_for_all_classes) - 1)],self.block_size_m)
        return naip_slice, nlcd_slice, lc_slice

    def next_minibatch(self, mb_size, patch_freq_per_class):
        features = np.zeros((mb_size, self.num_color_channels*self.channel_times,
            self.block_size, self.block_size), dtype=np.float32)
        #mao: /16
        landcover = np.zeros((mb_size, (self.stain_num+1)*self.channel_times,
            self.block_size_m, self.block_size_m), dtype=np.float32)
        lc_weight_map = np.zeros((mb_size, 1,
            self.block_size_m, self.block_size_m), dtype=np.float32)
        masks = np.zeros((mb_size, self.stain_num,
            self.block_size_m, self.block_size_m), dtype=np.float32)


        interval_centers = np.zeros(
            (mb_size, self.stain_num*self.channel_times),
            dtype=np.float32)
        interval_radii = np.zeros(
            (mb_size, self.stain_num*self.channel_times),
            dtype=np.float32)

        # Sample patches according to labels
        # sampler based on the number of training images from 2 lists
        if len(self.patches2) == 0:
            sample_from_new_list = False
        else:
            random_value=random.random()

            if random_value<= self.dots_patch_ratio
                print('random_value<= self.dots_patch_ratio')
                sample_from_new_list = True
            else:
                print('random_value > self.dots_patch_ratio')
                sample_from_new_list = False

        ins_id = 0
        while ins_id < mb_size:
            '''
            if len(self.patches2) == 0:
                sample_from_new_list = False
            #else:
            #    sample_from_new_list = True
            elif ins_id%6 <= 1:
                sample_from_new_list = True
            else:
                sample_from_new_list = False
            '''
            ########################################
            if sample_from_new_list:
                self.class_iter_idx=10
                dots_flag=1
                print('class_iter_idx0',self.class_iter_idx)
            elif ins_id%2 == 0 :
                self.class_iter_idx=np.random.choice(5)
                print('class_iter_idx1',self.class_iter_idx)
                self.class_iter_idx=2*self.class_iter_idx
                print('class_iter_idx2',self.class_iter_idx)
                dots_flag=0
            else:
                self.class_iter_idx+=1
                dots_flag=0
            if self.class_iter_idx>10:
                continue
            print('self.class_iter_idx',self.class_iter_idx)
            print('len(self.patch_list_for_all_classes)',len(self.patch_list_for_all_classes))
            print('len(self.patch_list_for_all_classes[10])',len(self.patch_list_for_all_classes[10]))
            for i in range(self.channel_times):
                if self.class_iter_idx!=10 and dots_flag==0:
                    naip_slice, nlcd_slice, lc_slice, nlcd_class_count = \
                        self.sample_slices_from_list(self.patch_list_for_all_classes[self.start_stain*2+self.class_iter_idx],self.block_size_m,)
                else:
                    naip_slice, nlcd_slice, lc_slice, nlcd_class_count = \
                        self.sample_slices_from_list_dotsimg(self.patch_list_for_all_classes[self.start_stain*2+self.class_iter_idx],self.block_size_m,)
                self.class_count_arr += np.array(nlcd_class_count)


                nlcd_slice=nlcd_slice[:,0,0]
                nlcd_dist1=nlcd_dist[:,0]*self.mu_times
                nlcd_dist1=nlcd_dist1[nlcd_slice>0]
                nlcd_var1=nlcd_var[:,0]*self.sigma_times
                nlcd_var1=nlcd_var1[nlcd_slice>0]
                interval_centers1= nlcd_dist1 #5 values
                interval_radii1 = nlcd_var1 #5 values

                if i==0:
                    naip_slice_stack=naip_slice
                    nlcd_slice_stack=nlcd_slice
                    lc_slice_stack=lc_slice
                    interval_centers_stack=interval_centers1
                    interval_radii_stack=interval_radii1

                else:
                    naip_slice_stack=np.concatenate([naip_slice_stack,naip_slice],axis=0)#3
                    nlcd_slice_stack=np.concatenate([nlcd_slice_stack,nlcd_slice],axis=0)#10
                    lc_slice_stack=np.concatenate([lc_slice_stack,lc_slice],axis=0)#6
                    interval_centers_stack=np.concatenate([interval_centers_stack,interval_centers1],axis=0)#5
                    interval_radii_stack=np.concatenate([interval_radii_stack,interval_radii1],axis=0)#5

            #naip_slice = color_aug(naip_slice)
            #for i in range(5):

            features[ins_id, :, :, :] = naip_slice_stack
            landcover[ins_id, :, :, :] = lc_slice_stack#self.to_one_hot(lc_slice, self.num_landcover_classes)
            lc_weight_map[ins_id, :, :, :] = 1.0
            #mask_temp=np.zeros((10,nlcd_slice.shape[1],nlcd_slice.shape[2]))
            '''
            for i in range(5):
                temp=self.to_one_hot(nlcd_slice[i,:,:], 2)
                mask_temp[i*2:i*2+2,:,:]=temp
            '''

            mask1=np.zeros((self.stain_num, self.block_size_m,self.block_size_m))
            if self.class_iter_idx!=10:
                mask1[int(np.floor(float(self.class_iter_idx)/2)),:,:]=1
            masks[ins_id, :, :, :] = mask1#nlcd_slice#self.to_one_hot(nlcd_slice, self.num_nlcd_classes)


            interval_centers[ins_id, :] = interval_centers_stack #5*channel_times values
            interval_radii[ins_id, :] = interval_radii_stack #5*channel_times values
            ins_id+=1



        self.print_class_count()
        result = {
            self.fsi: cntk.io.MinibatchData(cntk.Value(batch=features),
                        mb_size, mb_size, False),
            self.lsi: cntk.io.MinibatchData(cntk.Value(batch=landcover),
                        mb_size, mb_size, False),
            self.lwi: cntk.io.MinibatchData(cntk.Value(batch=lc_weight_map),
                        mb_size, mb_size, False),
            self.msi: cntk.io.MinibatchData(cntk.Value(batch=masks),
                        mb_size, mb_size, False),
            self.csi: cntk.io.MinibatchData(cntk.Value(batch=interval_centers),
                        mb_size, mb_size, False),
            self.rsi: cntk.io.MinibatchData(cntk.Value(batch=interval_radii),
                        mb_size, mb_size, False)
        }

        return result
