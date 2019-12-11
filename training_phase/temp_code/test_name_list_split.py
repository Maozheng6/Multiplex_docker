import os
def split_patch_per_class( all_patches):
    patch_list_for_all_classes = []
    stain_all=[]
    for nlcd_class in range(10):
        
        patch_list_per_class = [x for x in all_patches if \
                    os.path.basename(x).split('.npy')[0].split('_')[nlcd_class+1]=='1.0']
        print('List len {} for class {}'.format(len(patch_list_per_class), nlcd_class))
        patch_list_for_all_classes.append(patch_list_per_class)
    return patch_list_for_all_classes

patch_list='maozheng_patch_list_multiplex.txt'
all_patches = [line.rstrip('\n') for line in open(patch_list)]
new=split_patch_per_class(all_patches)
print(all_patches)
print(new)