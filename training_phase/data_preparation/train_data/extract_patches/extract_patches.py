import os
import numpy as np
from PIL import Image
from scipy.ndimage import filters
from skimage.segmentation import slic

# Map each label text to RGB color
text_2_rgb_id = {
    'cd16': [(0, 0, 0), 1],
    'cd20': [(255, 0, 0), 2],
    'cd3': [(255, 255, 0), 3],
    'cd4': [(10, 150, 255), 4],
    'cd8': [(200, 0, 200), 5],
    'k17': [(100, 80, 80), 6],
    'k17 neg': [(170, 170, 170), 7],
    'k17 - neg': [(170, 170, 170), 7],
    'k17-neg': [(170, 170, 170), 7],
    'k17-negative': [(170, 170, 170), 7],
    'k17 negative tumor': [(170, 170, 170), 7],
    'k17-negative tumor': [(170, 170, 170), 7],
    'k17-neg tumor': [(170, 170, 170), 7],
    'k17-neh': [(170, 170, 170), 7],
    'background': [(0, 100, 0), 8],
}

# Resize patches according to this factor
patch_resize_factor = 0.8448673587081891580161476355248

# Extract patches of this size from .tif tiles
patch_extract_dim = 400


# Patch extraction step size
patch_extract_step_size = 300

# Output patch folder
max_patch_num=1
output_folder = 'output_patches_60_wsi_0.84_max_patch_'+str(max_patch_num)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
# Load a single point file. Such as:
# L6745/Image_645.tif-points/Points 1.txt
def load_point(point_file_path):
    lines = [line.rstrip() for line in open(point_file_path)]

    # First line, second field
    label_text = lines[0].split('\t')[1].lower()
    if label_text not in text_2_rgb_id:
        return None, None
    label_rgb_id = text_2_rgb_id[label_text]

    xy_coords = np.array([[float(f) * patch_resize_factor for f in x.split()] for x in lines[3:]])

    return label_rgb_id, xy_coords

# Break image and labels to tiles
def break_to_patches(tif_path, im, im_labelvis, im_labelmat):
    output_prefix = tif_path[:-len('.tif')].replace('/', '-').replace('_', '-')
    print('cropppingggggggggggggggggggggggg')
    print(list(range(0, im.shape[0] - patch_extract_dim, patch_extract_step_size)) + [im. shape[0] - patch_extract_dim])
    print(list(range(0, im.shape[1] - patch_extract_dim, patch_extract_step_size)) +  [im.shape[1] - patch_extract_dim])

    for x in list(range(0, im.shape[0] - patch_extract_dim, patch_extract_step_size)) + [im.shape[0] - patch_extract_dim]:
        for y in list(range(0, im.shape[1] - patch_extract_dim, patch_extract_step_size)) + [im.shape[1] - patch_extract_dim]:
            patch = im[x : x + patch_extract_dim, y : y + patch_extract_dim, :]
            if patch.shape[0] != patch_extract_dim or patch.shape[1] != patch_extract_dim:
                print('patch extracted has incorrect size: {}x{}'.format(patch.shape[0], patch.shape[1]))
                continue
            Image.fromarray(patch).save('{}/DotsIM-{}-{}-{}.png'.format(output_folder, output_prefix, x, y))

            patch_labelvis = im_labelvis[x : x + patch_extract_dim, y : y + patch_extract_dim, :]
            Image.fromarray(patch_labelvis).save('{}/DotsVis-{}-{}-{}.png'.format(output_folder, output_prefix, x, y))

            patch_labelmat = im_labelmat[x : x + patch_extract_dim, y : y + patch_extract_dim]
            training_patch = np.concatenate((patch, patch_labelmat[..., np.newaxis]), axis=-1)
            np.save('{}/Dots-{}-{}-{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.npy'.format(
                output_folder, output_prefix, x, y,
                (patch_labelmat == 0).sum() / 200,
                (patch_labelmat == 1).sum() / 200,
                (patch_labelmat == 2).sum() / 200,
                (patch_labelmat == 3).sum() / 200,
                (patch_labelmat == 4).sum() / 200,
                (patch_labelmat == 5).sum() / 200,
                (patch_labelmat == 6).sum() / 200,
                (patch_labelmat == 7).sum() / 200,
                (patch_labelmat == 8).sum() / 200,
                ), training_patch)


# Process each tif and point folder. Such as:
# L6745/Image_645.tif L6745/Image_645.tif-points
def process_tif_and_points(tif_path, points_folder_path,basefolder):
    print('Doing', tif_path, points_folder_path)

    # The multiplex image
    # read png/tif
    im = Image.open(tif_path[:-4] + '.png').convert('RGB')
    print('im.shape',np.array(im).shape)
    im = im.resize((int(im.size[0] * patch_resize_factor), int(im.size[1] * patch_resize_factor)), Image.BICUBIC)
    print('im.shape resized',np.array(im).shape)
    im = np.array(im)
    # The label image for visualization
    im_labelvis = np.ones_like(im) * 255
    # The label matrix
    im_labelmat = np.ones((im.shape[0], im.shape[1]), dtype=np.uint8) * text_2_rgb_id['background'][1]

    # SLIC superpixel
    im_segments = slic(im, compactness = 10.0, n_segments = im.shape[0] * im.shape[1] / 200, sigma = 5)

    for point_file in os.listdir(points_folder_path):
        if not point_file.endswith('.txt'):
            continue
        point_file_path = os.path.join(points_folder_path, point_file)
        label_rgb_id, xy_coords = load_point(point_file_path)
        if label_rgb_id == None or len(xy_coords) == 0:
            continue

        label_rgb, label_id = label_rgb_id
        for x, y in xy_coords:
            int_x = int(x)
            int_y = int(y)
            # if points out of boundary a little
            if  int_y >= np.array(im_segments).shape[0] and int_y < np.array(im_segments).shape[0]+10:
                int_y = np.array(im_segments).shape[0] -1
            if  int_x >= np.array(im_segments).shape[1] and int_x < np.array(im_segments).shape[1]+10:
                int_x = np.array(im_segments).shape[1] -1
            seg_id = im_segments[int_y, int_x]
            im_labelvis[im_segments == seg_id, :] = label_rgb
            im_labelmat[im_segments == seg_id] = label_id

            #im_labelvis[int_y - 5 : int_y + 5, int_x - 5 : int_x + 5, :] = label_rgb
            #im_labelmat[int_y, int_x] = label_id

    im_label_path = tif_path[:-len('.tif')] + '_label.png'
    Image.fromarray(im_labelvis).save(im_label_path)

    tif_path1=tif_path[len(basefolder)::]
    print('tif_path1',tif_path1)
    break_to_patches(tif_path1, im, im_labelvis, im_labelmat)

def main():
    basefolder='/scratch/KurcGroup/mazhao/ICCV_NEW_DOTS_data_code_v2_regi-wsi/20190603_more_multiplex_labels/'
    patch_count=0
    for folder in os.listdir(basefolder):
        patch_count+=1
        if not os.path.isdir(basefolder+folder):
            continue
        if patch_count>max_patch_num:
            continue
        print(basefolder+folder)
        for tif in os.listdir(basefolder+folder):
            if not tif.endswith('tif'):
                continue
            #is png exist?
            #if not os.path.isfile(os.path.join(folder, tif[:-4] + '.png')):
            #    continue
            tif_path = os.path.join(basefolder+folder, tif)
            points_folder_path = tif_path + '-points'
            if not os.path.isdir(points_folder_path):
                continue
            print('tif_path',tif_path)
            print(points_folder_path)
            process_tif_and_points(tif_path, points_folder_path,basefolder)

if __name__ == '__main__':
    main()

