import os
import openslide
import glob
import json
files=glob.glob('../../multiplex-wsi/*multires.tif')
files1=glob.glob('../../multiplex-wsi/single_stain/*multires.tif')
files+=files1
print(files)
size_dict={}
for file_i in files:
    print(file_i)
    slide=openslide.OpenSlide(file_i)
    size1,size2=slide.level_dimensions[0]
    size_dict[os.path.basename(file_i)]=[size1,size2]

print(size_dict)

with open('slide_size.json', 'w') as f:
    json.dump(size_dict, f)
