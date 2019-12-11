import json
import os
import collections
import sys
import json
import glob
def gen_meta_json(in_path, image_id, wsi_width, wsi_height, method_description,
        save_folder,analysis_id,input_file_suffix,output_file_suffix):
    file_id = os.path.basename(in_path)[: -len(input_file_suffix)]
    fields = file_id.split('_')
    x = int(fields[0])
    y = int(fields[1])
    size1 = int(fields[2])
    size2 = int(fields[3])
    mpp = float(fields[4])

    dict_model = collections.OrderedDict()
    dict_model['input_type'] = 'wsi'
    dict_model['otsu_ratio'] = 0.0
    dict_model['curvature_weight'] = 0.0
    dict_model['min_size'] = 1#min_nucleus_size
    dict_model['max_size'] = 5#max_nucleus_size
    dict_model['ms_kernel'] = 0
    dict_model['declump_type'] = 0
    dict_model['levelset_num_iters'] = 0
    dict_model['mpp'] = mpp
    dict_model['image_width'] = wsi_width
    dict_model['image_height'] = wsi_height
    dict_model['tile_minx'] = x
    dict_model['tile_miny'] = y
    dict_model['tile_width'] = size1
    dict_model['tile_height'] = size2
    dict_model['patch_minx'] = x
    dict_model['patch_miny'] = y
    dict_model['patch_width'] = size1
    dict_model['patch_height'] = size2
    dict_model['output_level'] = 'mask'
    dict_model['out_file_prefix'] = file_id
    dict_model['subject_id'] = image_id
    dict_model['case_id'] = image_id
    dict_model['analysis_id'] = analysis_id
    dict_model['analysis_desc'] = '{}'.format(
            method_description)

    json_str = json.dumps(dict_model)

    fid = open(os.path.join(save_folder, file_id+output_file_suffix), 'w')
    print(os.path.join(save_folder, file_id+output_file_suffix))
    fid.write(json_str)
    fid.close()

def start_json(image_id,stain_idx,inpath,save_folder,method_prefix,analysis_id,output_folder_suffix,png_path,input_file_suffix,output_file_suffix,slide_suffix,input_folder_suffix):
    image_id = image_id[0:-len(input_folder_suffix)]
    print('image_id',image_id)
    method_description = method_prefix
    with open('slide_size.json', 'r') as f:
        size_dict = json.load(f)
    wsi_width, wsi_height = size_dict[image_id+slide_suffix]
    files=glob.glob(png_path)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    for in_path in files:
        print(in_path)
        print('save_folder',save_folder)
        gen_meta_json(in_path, image_id+output_folder_suffix, wsi_width, wsi_height, method_description,
                save_folder,analysis_id,input_file_suffix,output_file_suffix)
