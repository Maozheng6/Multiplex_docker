import cv2
import glob
import os
slide_name = 'N9430-multires'
inputfolder='/scratch/KurcGroup/mazhao/tiles_slide/'+slide_name+'/'
files=glob.glob(inputfolder+'/*SEG.png')
save_folder = '/scratch/KurcGroup/mazhao/tiles_slide/'+slide_name+'_RGB/'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
print('len(files)',len(files))
count=0

for file_i in files[0:10]:
    print(file_i)
    count+=1
    print(count/len(files))
    #img=cv2.imread(file_i)
    #img=img[:,:,::-1]
    #cv2.imwrite(file_i,img)
    #cv2.imshow(img)

