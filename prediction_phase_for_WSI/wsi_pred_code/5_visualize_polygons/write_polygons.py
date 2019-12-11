import numpy as np
import cv2
import os
def write_polygons(argmax_name,stain_index):
    save_path = '.'
    global_xy_offset= [int(x) for x in os.path.basename(argmax_name).split('_')[0:2]]
    argmax_map = np.load(argmax_name)
    binary_mask = np.zeros((argmax_map.shape[0],argmax_map.shape[1])).astype('uint8')
    binary_mask[argmax_map == stain_index+1]=255
    img = binary_mask
    img = cv2.resize(img,(4000,4000))
    cv2.imwrite(os.path.join(save_path,os.path.basename(argmax_name)[0:-len('.npy')]+'-binary.png'),img)
    cv2.imwrite(os.path.join(save_path,os.path.basename(argmax_name)[0:-len('.npy')]+'-binary-thumb.png'),cv2.resize(img,(500,500)))

    #print(np.max(img),img.shape)
    #poly = cv2.findContours(img.astype('uint8'), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    poly = cv2.findContours(img.astype('uint8'), cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    contour,hia=poly
    draw_contour=np.zeros((img.shape[1],img.shape[0],3)).astype('uint8')

    draw_contour[:,:,0]=img
    draw_contour[:,:,1]=img
    draw_contour[:,:,2]=img
    draw_contour=cv2.drawContours(draw_contour, contour, -1, (0,255,0), 3)
    cv2.imwrite('draw_contour.png',draw_contour)
    num_contour=len(contour)
    file_id=os.path.basename(argmax_name)[0:-len('.npy')]
    fid = open(os.path.join(save_path,file_id+'-features.csv'), 'w')
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

#argmax_name ='/scratch/KurcGroup/mazhao/wsi_prediction/pred_out_iccv_resized_300/O3936_6.1/116001_80001_4000_4000_0.25_1_SEG_argmax.npy'
#'/scratch/KurcGroup/mazhao/quip4_poly_dots_model_resized/transfered10_300_no-hierar_argmax_maps/O3936-multires/116001_80001_4000_4000_0.25_1.npy'
#stain_index = 6
#write_polygons(argmax_name,stain_index)
