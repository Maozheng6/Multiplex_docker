import cv2
import subprocess
import os
import numpy as np
#file_name='/scratch/KurcGroup/mazhao/wsi_tiles_prediction/O3936-multires/96001_72001_4000_4000_0.25_1_SEG_0_pred.png'
#save_path='/scratch/KurcGroup/mazhao/quip4_files/'+os.path.basename(os.path.dirname(file_name)+'/'+cell_type[stain_num])
#save_path='.'
def get_poly(pair):
        thre_mode = 0
        print('len(pair)',len(pair))
        file_name,save_path,stain_index,argmax_name,input_file_suffix, output_file_suffix = pair
        print(pair)
        if argmax_name==None:
            thre_mode = 1
        else:
            print('argmax mode!')
        #file_name is the heatmap absolute path,save_path is the folder to save the result
        #if not os.path.isfile(os.path.join(save_path,file_id+'-features.csv')):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        global_xy_offset= [int(x) for x in os.path.basename(file_name).split('_')[0:2]]
        if thre_mode==1:
            img=cv2.imread(file_name,0)
            print('file_name',file_name)
            thre,img=cv2.threshold(img,210,255,cv2.THRESH_BINARY)
        else:
            if argmax_name.endswith('png'):
                argmax_map = cv2.imread(argmax_name,0)#np.load(argmax_name)
            elif argmax_name.endswith('npy'):
                argmax_map = np.load(argmax_name)
            binary_mask = np.zeros((argmax_map.shape[0],argmax_map.shape[1])).astype('uint8')
            binary_mask[argmax_map == stain_index+1]=255
            img = binary_mask
            #resizing to 2 times!!!!!!!!!!!!!!!!!
            #heat_map = cv2.imread(file_name)
            #img = cv2.resize(img,(heat_map.shape[1],heat_map.shape[0]),cv2.INTER_NEAREST)
            #cv2.imwrite(os.path.join(save_path,os.path.basename(file_name)[0:-10]+'-binary.png'),img)

        #print(np.max(img),img.shape)
        #poly = cv2.findContours(img.astype('uint8'), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        poly = cv2.findContours(img.astype('uint8'), cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        contour,hia=poly
        num_contour=len(contour)
        file_id=os.path.basename(file_name)[0:-len(input_file_suffix)]
        fid = open(os.path.join(save_path,file_id+output_file_suffix), 'w')
        fid.write('AreaInPixels,PhysicalSize,Polygon\n')
        for idx in range(num_contour):
            contour_i = contour[idx]
            physical_size = cv2.contourArea(contour_i)
            #print(physical_size)
            #if physical_size>4000 or physical_size<5:
            #    continue
            contour_i = contour_i[:,0,:].astype(np.float32)

            contour_i[:, 0] = contour_i[:, 0] + global_xy_offset[0]

            contour_i[:, 1] = contour_i[:, 1]  + global_xy_offset[1]
            poly_str = ':'.join(['{:.1f}'.format(x) for x in contour_i.flatten().tolist()])
            #print(poly_str)
            fid.write('{},{},[{}]\n'.format(
		int(physical_size), int(physical_size), poly_str))
        fid.close()
        return 1
