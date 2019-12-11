import cv2
import os
ratio=32/(float(2013*2)/240)
#ratio=1.9345
folder='./TCGA_masks/'#'./TCGA_masks/'#
save_folder='./TCGA_masks_resized'+str(ratio)+'/'#'./TCGA_masks_resized/'#
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

files=os.listdir(folder)
count=0

#filelist=open('TCGA_test_slides.txt','w')
for i in files:
    count+=1
    print(count/len(files))
    #i=i[0:-4]
    #filelist.write(i)
    #filelist.write('\n')
    print(os.path.join(folder,i))
    img=cv2.imread(os.path.join(folder,i))
    resized=cv2.resize(img,(int(img.shape[1]*ratio),int(img.shape[0]*ratio)))#,interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(save_folder,i),resized)
    