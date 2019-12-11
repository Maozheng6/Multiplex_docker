import os
import glob
import cv2
from get_poly import get_poly
infolder='../wsi_tiles_prediction/'
save_folder='../quip4_poly_dots_model_resized_argmax_maps/'
folders=os.listdir(infolder)
cell_type={2:'CD3-Double_Negative_T_cell-Yellow',0:'CD16-Myeloid_cell-Black',4:'CD8-Cytotoxic_cell-Purple',1:'CD20-B_cell-Red',3:'CD4-Helper_T_cell-Cyan',5:'K17_Pos',6:'K17_Neg'}

folders.reverse()
for folder_i in folders:
    for stain in cell_type.keys():
        save_path = os.path.join(save_folder,folder_i,cell_type[stain])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        files = glob.glob(os.path.join(infolder,folder_i,'*'+str(stain)+'*'))
        for file_i in files:
            print(file_i)
            file_id=os.path.basename(file_i)[0:-10]
            if not os.path.isfile(os.path.join(save_path,file_id+'-features.csv')):
                get_poly(file_i,save_path)

