import glob
import os
import pandas as pd
input_folder='/scratch/KurcGroup/mazhao/tiles_slide/'
output_folder='/scratch/KurcGroup/mazhao/wsi_prediction_png/pred_out_6_slides_300/'
csv_folder = '/scratch/KurcGroup/mazhao/wsi_pred_code/2_generating_patch_list_csv/patch_lists_csv/'
suffix='_20x10x_comb/'
tiles=['3908','O6218_MULTI_3','P528']
#['N4277', 'L6745',  'N22034',  'N9430',  'O0135',  'O3936']
#tiles=['N3908_CD8_PURPLE_8','N3908_CD16_BLACK_10','N3908_K17_BROWN_9','N3908_CD20_RED_6', 'N3908_CD3_YELLOW_5','N3908_CD4_BLUE_7','O6218_CD3_YELLOW_7','O6218_CD16_BLACK_10','O6218_CD20_RED_5','O6218_CD4_BLUE_6','O6218_CD8_PURPLE_8','O6218_K17_BROWN_9']
#['3908','O6218_MULTI_3']
#['3908',   'L6745',  'O0135',   'O3936']
#['3908', 'L6745',  'N22034',  'N9430',  'O0135',  'O3105',  'O3936',  'O6218_MULTI_3']
for tile in tiles:
    csv_file=csv_folder+tile+'-multires.csv'
    input_df = pd.read_csv(csv_file)
    pair_list = input_df[["patch_path", "label"]].values
    #number of patches from wsi
    num_in_csv = len(pair_list)
    num_wsi_patches=len(glob.glob(input_folder+tile+'-multires'+'/'+'*png'))#*(8*2+1)
    #number of prediction
    num_pred=len(glob.glob(output_folder+tile+suffix+'/'+'*argmax.png'))
    #print(pair_list[:][1])
    print('')
    print(tile)
    print('num_wsi_patches,num_pred,num_in_csv')
    print(num_wsi_patches,num_pred,num_in_csv)
    if num_wsi_patches!=num_pred or num_in_csv!=num_wsi_patches:
        print('{0} has {1:d}/{2:d} = {3:05.3f} been done.'.format(tile,num_pred,num_wsi_patches,num_pred/num_wsi_patches))
    else:
        print(tile,'is all {0}/{1}={2:05.3f} done!!!'.format(num_pred,num_wsi_patches,num_pred/num_wsi_patches))

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
