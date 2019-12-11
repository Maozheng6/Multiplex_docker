import os
import shutil
infolder1='/home/lehhou/maozheng/LYM_SR2_tumor_60output_newstd/binarized_prediction_without_holes_parallel/edge_overlay/'
infolder2='/home/lehhou/maozheng/LYM_SR2_tumor_60output_newstd/binarized_prediction_with_holes_parallel/edge_overlay/'
files=os.listdir(infolder1)
save_folder='picked_results_remholes'
folder1='rmholes'
folder2='ori'
if not os.path.exists(os.path.join(save_folder,folder1)):
    os.makedirs(os.path.join(save_folder,folder1))
if not os.path.exists(os.path.join(save_folder,folder2)):
    os.makedirs(os.path.join(save_folder,folder2))
count=0
for name in files:
    if count<10:
        if name.endswith('overlay_edge.png') and os.path.isfile(os.path.join('/home/lehhou/maozheng/LYM_SR2_tumor_60output_newstd/binarized_prediction_with_holes_parallel/edge_overlay/',name)):
            count+=1
            shutil.copyfile(os.path.join(infolder1,name),os.path.join(save_folder,folder1,name))
            shutil.copyfile(os.path.join(infolder2,name),os.path.join(save_folder,folder2,name))
    else:
       exit()