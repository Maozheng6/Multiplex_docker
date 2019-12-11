import os
import time
import glob
import concurrent.futures
import cv2
from get_poly_6slides import get_poly
import numpy as np
import sys

slide_idx = sys.argv[1]#[0:-len('-multires.tif')]
pred_suffix = sys.argv[2]

pred_folder_name = slide_idx+pred_suffix
print('pred_folder_name',pred_folder_name)

#infolder='/scratch/KurcGroup/mazhao/wsi_prediction/pred_out_6_slides_300/'+pred_folder_name
#save_folder='../../quip4_poly_dots_model_resized/6_slides/'

infolder ='/scratch/KurcGroup/mazhao/wsi_prediction/pred_out_O0135_fixbgr_500/'+pred_folder_name
save_folder='../../quip4_poly_dots_model_resized/6_slides_comp_v2/'
input_path_suffix = '_6.6_1.0'
input_file_suffix = '_1_SEG_argmax.png'
save_folder_suffix = '-multires'
print('infolder',infolder)
#argmax_save_folder='../quip4_poly_dots_model_resized/transfered10_300/'

#folders=glob.glob(infolder)
folders = [infolder]
cell_type_with_BG={2:'CD3-Double_Negative_T_cell-Yellow',0:'CD16-Myeloid_cell-Black',4:'CD8-Cytotoxic_cell-Purple',1:'CD20-B_cell-Red',3:'CD4-Helper_T_cell-Cyan',5:'K17_Pos',6:'K17_Neg',7:'BG'}

cell_type={2:'CD3-Double_Negative_T_cell-Yellow',0:'CD16-Myeloid_cell-Black',4:'CD8-Cytotoxic_cell-Purple',1:'CD20-B_cell-Red',3:'CD4-Helper_T_cell-Cyan',5:'K17_Pos',6:'K17_Neg'}


make_list = []

folder_i = infolder
folder_i_base = os.path.basename(folder_i)
print('folder_i_base',folder_i_base)
#if not (folder_i_base.startswith('N22034') or folder_i_base.startswith('N9430')):
#    continue

files = glob.glob(os.path.join(folder_i,'*'+'argmax.npy'))
#argmax_save=None
#os.path.join(argmax_save_folder[0:-len('/')]+'_argmax_maps',folder_i+'-multires')
#if not os.path.exists(argmax_save):
#    os.makedirs(argmax_save)
for file_i in files:
    make_list.append([file_i,folder_i])
print('make_list',make_list)

'''
for folder_i in folders:
    #debug!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #selecting slides
    folder_i_base = os.path.basename(folder_i)
    print('folder_i_base',folder_i_base)
    if not (folder_i_base.startswith('N22034') or folder_i_base.startswith('N9430')):
        continue

    files = glob.glob(os.path.join(folder_i,'*'+str(0)+'.png'))
    #argmax_save=None
    #os.path.join(argmax_save_folder[0:-len('/')]+'_argmax_maps',folder_i+'-multires')
    #if not os.path.exists(argmax_save):
    #    os.makedirs(argmax_save)
    for file_i in files:
        make_list.append([file_i,folder_i])
'''
class make_pair_list():
    def __init__(self):
        self.pair_list=[]

    def make_argmax_and_list_for_one(self,pair):
        if True:

            file_i,folder_i = pair
            argmax_name = os.path.join(folder_i,os.path.basename(file_i)[0:-len('.png')]+'.npy')

            print('argmax_name',argmax_name)
            #os.path.join(argmax_save,os.path.basename(file_i)[0:-len('_SEG_0.png')]+'.npy')
            #generate argmax map
            '''
            if not os.path.exists(argmax_name):
                print('argmax_name',argmax_name)
                heat_0=cv2.imread(file_i)
                heat_stack=np.zeros((heat_0.shape[0],heat_0.shape[1],8))
                for stain in cell_type_with_BG.keys():
                    heat_i=cv2.imread(file_i[0:-len('0.png')]+str(stain)+'.png',0)
                    heat_stack[:,:,stain] = heat_i
                argmax_map = np.argmax(heat_stack,axis=-1)+1
                np.save(argmax_name,argmax_map)
            '''


            for stain in cell_type.keys():
                ############################
                #argmax process

                #binary_mask = np.zeros((heat_0.shape[0],heat_0.shape[1],3)).astype('uint8')
                #binary_mask[argmax_map == stain+1,:]=255
                #cv2.imwrite(save_path,file_prefix+'_'+str(stain)+'-binary.png')
                ######################################


                save_path = os.path.join(save_folder,os.path.basename(folder_i)[0:-len(input_path_suffix)]+save_folder_suffix,cell_type[stain])


                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                file_prefix=os.path.basename(file_i)[0:-len(input_file_suffix)]
                #cv2.imwrite(os.path.join(save_path,file_prefix+'_'+str(stain)+'-binary.png'),binary_mask)
                #if stain==0:
                #    cv2.imwrite(os.path.join(save_path,file_prefix+'_'+str(stain)+'-argmax.png'),argmax_map)

                if  not os.path.isfile(os.path.join(save_path,file_prefix+'-features.csv')):
                    print(os.path.isfile(os.path.join(save_path,file_prefix+'-features.csv')))
                    print(os.path.join(save_path,file_prefix+'-features.csv'))
                    self.pair_list.append([file_i,save_path,stain,argmax_name])
                    print('len(self.pair_list)',len(self.pair_list))
                else:
                    print(os.path.join(save_path,file_prefix+'-features.csv'),'exists______________')
                    print(os.path.isfile(os.path.join(save_path,file_prefix+'-features.csv')))

                    #get_poly(file_i,save_path)
def main():
    MAKE_PAIR=make_pair_list()
    #with concurrent.futures.ProcessPoolExecutor( max_workers=10) as executor:
    #    for  prime in  executor.map(MAKE_PAIR.make_argmax_and_list_for_one, make_list, chunksize=10):
    #        print(' is prime: %s' % ( prime))

    print('len(make_list)',len(make_list))
    for pp in make_list:
        MAKE_PAIR.make_argmax_and_list_for_one(pp)
    print('len(pair_list)',len(MAKE_PAIR.pair_list))
    print('len(make_list)',len(make_list))
    with concurrent.futures.ProcessPoolExecutor( max_workers=30) as executor:
        for number, prime in zip(MAKE_PAIR.pair_list, executor.map(get_poly, MAKE_PAIR.pair_list, chunksize=4)):
            print('%s is prime: %s' % (number, prime))


def main_none_para():
    #fid=open(save_folder+'/'+'rm_log.txt','w')
    for pair in pair_list:
        a=get_poly(pair)
if __name__ == '__main__':
    main()
    #main_none_para()
