import os
folder1='/scratch/KurcGroup/mazhao/wsi_tiles_prediction/'
folder2='/gpfs/home/mazhao/wsi_tiles_prediction/'
tiles=['3908', 'L6745',  'N22034',  'N9430',  'O0135',  'O3105',  'O3936',  'O6218_MULTI_3']
for tile in tiles:
    num1=len(os.listdir(folder1+tile+'-multires'))
    num2=len(os.listdir(folder2+tile+'-multires'))
    if num1!=num2:
        print(tile,num1,num2)
    else:
        print('GOOD',tile)
