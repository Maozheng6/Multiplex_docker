
import os

import glob 
import shutil
import cv2
import copy
import matplotlib.pyplot as plt
import concurrent.futures

files=glob.glob(input_folder+'/*.zip')

results='/mnt/blobfuse/pred-output/ByMZ-Multi_6layer_99_1-2-0.1-1_1_1_1_1_1_1_1_1_1-0.0005-0.9995-stain5-mu1.0-sigma1.0-start_stain0-GPU1/'

annotators=['Areeha','Christian','Emily','Inga']
base_folder='/mnt/blobfuse/train-output/ByMZ/data_dots_labels_for_multiplex'
unzipped_folder='unzipped'#Areeha/2938_cd20h_cd3h_cd4h_cd8h.png-points/'
image_folder='images'#Areeha/2938_cd20h_cd3h_cd4h_cd8h.png'
save_folder='dots_visualization'#Areeha/'
annotator='Inga'
'''
#unzip
for file_i in files:
    if file_i.endswith('.zip'):
        print('unzip '+file_i+' '+os.path.join(base_folder,unzipped_folder,annotator)+'/'+os.path.basename(file_i)[0:-4])
        if not os.path.exists(os.path.join(base_folder,unzipped_folder,annotator)+'/'+os.path.basename(file_i)[0:-4]):
            os.makedirs(os.path.join(base_folder,unzipped_folder,annotator)+'/'+os.path.basename(file_i)[0:-4])
        shutil.copyfile(file_i,os.path.join(os.path.join(base_folder,unzipped_folder,annotator),os.path.basename(file_i)))
        os.system('unzip '+os.path.join(base_folder,unzipped_folder,annotator)+'/'+os.path.basename(file_i))
        os.system('cp *.txt '+os.path.join(base_folder,unzipped_folder,annotator)+'/'+os.path.basename(file_i)[0:-4])
        os.system('rm *.txt')
'''


def parse_txt_to_color_and_corrdi(file_i):
    txt_file=open(file_i,'r')
    lines = list(txt_file)
    color=lines[0].rstrip('\n').split('\t')[-1]
    dots_num=int(lines[2].rstrip('\n').split('\t')[-1])
    cordinates_line=lines[3:]
    cordinates_list=[]
    for cord in cordinates_line:
        cord_list=cord.rstrip('\n').split('\t')
        cord_list=[round(float(x)) for x in cord_list ]
        cordinates_list.append(cord_list)
    print(color,cordinates_list)
    if color.endswith('Color'):
        color=color[0:-5]
    
    return color,dots_num,cordinates_list
    


if not os.path.exists(save_folder):
    os.makedirs(save_folder)
def visualize_one_patch(folder_i,imname,save_folder,colors_by_one_annotator):
    files_in_i=glob.glob(folder_i+'/*.txt')
    idx=0
    f, axarr = plt.subplots(1, len(files_in_i)+1,dpi=1000)#plt.subplots
    color_indx={'Black':0,'Red':1,'Yellow':2,'Cyan':3,'Purple':4,'Blue':3,'Green':2}
    for file_i in files_in_i:
        print('%',file_i)
        color,dots_num,cordinates_list=parse_txt_to_color_and_corrdi(file_i)
        if color not in colors_by_one_annotator:
            colors_by_one_annotator.append(color)
        img=cv2.imread(os.path.join(results,imname[0:-4]+'_'+str(color_indx[color])+'_overlay.png'))
        print(img.shape)
        ori_image=copy.deepcopy(img)
        if len(cordinates_list)>0:
            for center in cordinates_list:
                img_dot=cv2.circle(img, (center[0],center[1]), 0, color=[0,255,0], thickness=5)
        else:
            img_dot=img
        if idx==0:
            axarr[idx].imshow(ori_image[:,:,::-1]/255)
            axarr[idx].set_title('RGB',fontsize=6)
            axarr[idx].axis('off')
        axarr[idx+1].imshow(img_dot[:,:,::-1]/255)
        axarr[idx+1].set_title(color+' : '+str(dots_num),fontsize=6)
        axarr[idx+1].axis('off')
        idx+=1
    f.savefig(os.path.join(save_folder,os.path.basename(imname)), bbox_inches='tight', pad_inches=0)
    plt.close(f)
    return colors_by_one_annotator
annotators=['Areeha','Christian','Emily','Inga']
colors_by_annotators={}
#for annotator in annotators:
def process_one_annotator(annotator):
    im_names=glob.glob(os.path.join(base_folder,image_folder,annotator)+'/*.png')
    colors_by_one_annotator=[]
    for imname in im_names:
        #print(imname)
        dots_folder=os.path.join(base_folder,unzipped_folder,annotator,os.path.basename(imname)+'-points')
        print(dots_folder,os.path.exists(dots_folder))
        save_dir=os.path.join(base_folder,save_folder,annotator)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        colors_by_one_annotator=visualize_one_patch(dots_folder,imname,save_dir,colors_by_one_annotator)
    colors_by_annotators[annotator]=colors_by_one_annotator



def main():
    with concurrent.futures.ProcessPoolExecutor( max_workers=4) as executor:
        for number, prime in zip(annotators, executor.map(process_one_annotator, annotators, chunksize=1)):
            print('%s is prime: %s' % (number, prime))


if __name__ == '__main__':
    main()
    print(colors_by_annotators)