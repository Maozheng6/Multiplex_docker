import glob
import os
import pandas as pd
folder1='/scratch/KurcGroup/mazhao/tiles_slide/'
folder2='/scratch/KurcGroup/mazhao/wsi_prediction/pred_out_iccv_resized_300/'

tiles=['3908',   'N9430',  'O0135',   'O3936']
#['3908', 'L6745',  'N22034',  'N9430',  'O0135',  'O3105',  'O3936',  'O6218_MULTI_3']
for tile in tiles:
    csv_file=tile+'-multires.csv'
    input_df = pd.read_csv(csv_file)
    pair_list = input_df[["patch_path", "label"]].values
    num3 = len(pair_list)
    num1=len(glob.glob(folder1+tile+'-multires'+'/'+'*png'))#*(8*2+1)
    num2=len(glob.glob(folder2+tile+'/'+'*fig.png'))
    #print(pair_list[:][1])
    print('num tiles','num fig', 'num csv',num1,num2,num3)
    if num1!=num2 or num3!=num1:
        print(tile,num1,num2,'%.3f'%(num2/num1))
        for file_i in glob.glob(folder1+tile+'-multires'+'/'+'*png'):
            file_base=os.path.basename(file_i)
            if not os.path.isfile(folder2+tile+'/'+file_base[0:-len('.png')]+'_8fig.png'):
                pass
                #print('not exist:',folder2+tile+'/'+file_base[0:-len('.png')]+'_8fig.png')

    else:
        print('GOOD',tile)

    for i in range(len(pair_list)):
        file_name = pair_list[i][0]
        if not os.path.isfile( file_name ):
            print('not exists',file_name)
    names=[x[0] for x in pair_list ]
    #else:
            #print('----------')
    for name in names:
        if not os.path.isfile(name):
            print('not exists',name)
