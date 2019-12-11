import os
import time
import glob
import concurrent.futures
import cv2
from get_poly import get_poly

infolder='../wsi_prediction/pred_out_iccv_300/'
save_folder='../quip4_poly_dots_model/'
folders=os.listdir(infolder)
cell_type={2:'CD3-Double_Negative_T_cell-Yellow',0:'CD16-Myeloid_cell-Black',4:'CD8-Cytotoxic_cell-Purple',1:'CD20-B_cell-Red',3:'CD4-Helper_T_cell-Cyan',5:'K17_Pos',6:'K17_Neg'}

pair_list=[]

for folder_i in folders:
    for stain in cell_type.keys():
        save_path = os.path.join(save_folder,folder_i+'-multires',cell_type[stain])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        files = glob.glob(os.path.join(infolder,folder_i,'*'+str(stain)+'.png'))

        for file_i in files:

            #print(file_i)
            file_prefix=os.path.basename(file_i)[0:-len('_SEG_1.png')]
            #file_i and file_id are too similar, variable should be as different as poosible, file_id can be file_prefix, be precise!!!
            #pair_list.append([file_i,save_path])
            if not os.path.isfile(os.path.join(save_path,file_prefix+'-features.csv')):
                print(os.path.isfile(os.path.join(save_path,file_prefix+'-features.csv')))
                print(os.path.join(save_path,file_prefix+'-features.csv'))
                print(len([file_i,save_path,None]),[file_i,save_path,None])
                pair_list.append([file_i,save_path,stain,None])
            else:
                print('exists______________')
                print(os.path.isfile(os.path.join(save_path,file_prefix+'-features.csv')))

            #    get_poly(file_i,save_path)
print('len(pair_list)',len(pair_list))
time.sleep(2)
def main():
    with concurrent.futures.ProcessPoolExecutor( max_workers=1) as executor:
        for number, prime in zip(pair_list, executor.map(get_poly, pair_list, chunksize=10)):
            print('%s is prime: %s' % (number, prime))


def main_none_para():
    #fid=open(save_folder+'/'+'rm_log.txt','w')
    for pair in pair_list:
        a=get_poly(pair)
if __name__ == '__main__':
    #main()
    main_none_para()
