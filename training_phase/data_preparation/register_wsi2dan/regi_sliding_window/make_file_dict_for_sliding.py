import os
import glob

Dan_folder_list = ['L28352_2', 'N22800_20', 'N5039_5', 'O3936_2', 'P28191_80', 'L29978_2', 'N24178_80', 'N8033_95', 'O5498_5', 'P29193_65', 'L6745_0', 'N24852_5', 'N8945_50', 'O6218_30', 'P2919_20', 'M036_15', 'N27093_95', 'N9430_90', 'O8372_10', 'P304_80', 'M28417_2', 'N27243_5', 'O0135_95', 'P0533_50', 'P31681_40_ariah', 'M3171_100', 'N27702_20', 'O21073_30', 'P0992_80', 'P4211_0', 'M3669_50', 'N28217_2_ariah', 'O21747_95', 'P22034_2', 'P528_30_ariah','M4213_2', 'N29055_30', 'O31013_80_ariah',
        'P23541_30', 'P670_5','N0982_10_ariah', 'N3908_70', 'O3105_10', 'P24146_90', 'N22034_90', 'N4277_0', 'O31619_0', 'P24230_20']

Dan_folder_list_index = [x.split('_')[0] for x in Dan_folder_list]

Dan_folder_list_index = [x[1:] for x in Dan_folder_list_index]

wsi_path ='/gpfs/scratch/mazhao/multiplex-wsi/'

wsi_files = glob.glob(wsi_path+'*.tif')

wsi_files = [os.path.basename(x) for x in wsi_files]

print(Dan_folder_list_index)
print(wsi_files)

wsi_files_index = [x.split('-')[0] for x in wsi_files]

wsi_files_index = [x[1:] for x in wsi_files_index]

print(wsi_files_index)

import pandas as pd

input_df = pd.read_csv('wsi_dan_map.csv')
pair_list = input_df[["wsi", "dan"]].values

print(pair_list)

wsi_dan_dict = {}
for x in pair_list:
    wsi_dan_dict[x[0]] = x[1]

print(wsi_dan_dict)















