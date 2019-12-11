import os
import shutil
import glob
wsi2dan_slide ={'M28417-multires': 'M28417_2', 'M4213-multires': 'M4213_2', 'N24852-multires': 'N24852_5', 'N27243-multires': 'N27243_5',   'N29055-multires': 'N29055_30', 'N5039-multires': 'N5039_5', 'O21747-multires': 'O21747_95', 'O8372-multires': 'O8372_10', 'P0992-multires':    'P0992_80', 'P24146-multires': 'P24146_90', 'P28191-multires': 'P28191_80', 'P304-multires': 'P304_80', 'P4211-multires': 'P4211_0', 'P670-     multires': 'P670_5', 'M3669-multires': 'M3669_50','N24178-multires':
        'N24178_80', 'N27093-multires': 'N27093_95', 'N27702-multires':            'N27702_20', '3908-multires': 'N3908_70', 'N8945-multires': 'N8945_50', 'O6218_MULTI_3-multires': 'O6218_30', 'P0533-multires': 'P0533_50',     'N22034-multires': 'P22034_2', 'P24230-multires': 'P24230_20', 'P29193-multires': 'P29193_65', 'P31681-multires': 'P31681_40', 'P528-multires': 'P528_30'}
#{'N22800-multires':'L22800_20', 'L28352-multires':'L28352_2', 'L29978-     multires':'L29978_2', 'M036-multires':'M036_15','P24146-multires':'P24146_90',  'N8032-multires':'_N8033_95'}
# {'O3936-multires':'O3936','L6745-multires':'L6745','N22034-multires':'N22034_90_Scale_bar_is_set_wrong', 'O0135-multires':'O0135','O3105-multires':      'O3105_10','N9430-multires':'N9430'}#'O3936-multires':'O3936',
for slide_name in wsi2dan_slide.keys():
    ref_folder = '/scratch/KurcGroup/mazhao/ICCV_NEW_DOTS_data_code_v2_regi-wsi/20190603_to_register/'+wsi2dan_slide[slide_name]+'/'
    dan_imgs = glob.glob(ref_folder+'*.tif')
    for file_i in dan_imgs:
        shutil.copyfile(file_i,'./regi_out_refined_4000-0.30_20190603_to_register/'+slide_name+'_'+os.path.basename(file_i)[0:-4]+'_dan.png')
