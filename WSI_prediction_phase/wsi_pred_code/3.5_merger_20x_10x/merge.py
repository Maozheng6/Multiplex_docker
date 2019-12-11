import numpy
import matplotlib.pyplot as plt
#import scipy.misc as misc
from PIL import Image
from numpy import array, newaxis, expand_dims
from collections import defaultdict
import numpy as np
from skimage import io
import cv2
import PIL
import os
import gc
from PIL import Image
import seaborn as sns; sns.set()
import copy
import sys
path_20x = sys.argv[2]#'/scratch/KurcGroup/mazhao/wsi_prediction_png/pred_out_6_slides_300/O6218_CD16_BLACK_10_6.1_20x/'
path_10x = sys.argv[1]#'/scratch/KurcGroup/mazhao/wsi_prediction_png/pred_out_6_slides_300/O6218_CD16_BLACK_10_6.6_10x/'
out_path = sys.argv[3]#'/scratch/KurcGroup/mazhao/wsi_prediction_png/pred_out_6_slides_300/O6218_CD16_BLACK_10_20x10x_comb/'
print('sys.argv 1-3', sys.argv[1],sys.argv[2],sys.argv[3])
if not os.path.exists(out_path):
    os.makedirs(out_path)
maximum = 0
thres_1 = 0.0
thres_2 = 0.0
remove_small_thre = 0
org_files = []
def MyFn(s):
    #print(s.split('/')[-1].split('_')[1])
    #print("check function")
    #print(s)
    #print(int(s.split('/')[-1].split('_')[0]))
    #print(int(s.split('/')[-1].split('_')[1]))
    return (int(s.split('/')[-1].split('_')[0]),int(s.split('/')[-1].split('_')[1]))

for r, d, f in os.walk(path_20x):
    for file in f:
        if 'argmax.png' in file:
            org_files.append(os.path.join(r, file))
print('##############################')
print('org_files',org_files)
for i in org_files:
	print (i)
	x = i.split('/')[-1].split('_')[0]
	y = i.split('/')[-1].split('_')[1]
	start = x+'_'+y
	print(start)
	a10x_files = []
	print("problem")
	print(path_10x)
	a10xfile = [filename for filename in os.listdir(path_10x) if filename.startswith(start) and filename.endswith('argmax.png')]
	print("to take")
	print(i)
	print(a10xfile[0])
	path_10x_path = path_10x+a10xfile[0]
	print(path_10x)
	numpy20x = cv2.imread(i[0:-3]+'png',0)#numpy.load(i)
	numpy10x = cv2.imread(path_10x_path[0:-3]+'png',0)#numpy.load(path_10x_path)
	print("numpy 10x shape")
	print(path_10x_path)
	print(numpy10x.shape)
	print("numpy 20x shape")
	print(i)
	print(numpy20x.shape)
#	if(numpy20x.shape!=numpy10x.shape):
#`		print("not smae")
#		break
#	continue
	#if(numpy10x.shape != (4000,4000)):
	#	numpy10x = cv2.resize(numpy10x.astype('uint8'),(4000,4000),interpolation=cv2.INTER_LINEAR )
	if(numpy20x.shape!= numpy10x.shape):
		numpy10x = cv2.resize(numpy10x.astype('uint8'),numpy20x.shape,interpolation=cv2.INTER_LINEAR )
	new_numpy = np.full(numpy10x.shape, 8, dtype=int)
	mask_20 = (numpy20x<6).astype(int)
	new_numpy = new_numpy + numpy.multiply(mask_20,numpy20x) - numpy.multiply(new_numpy,mask_20)
	new_numpy = new_numpy.astype(int)
	print("check")
	print(mask_20)
	mask_20_inv =copy.deepcopy( mask_20)
	mask_20_inv[mask_20_inv == 1] = -1
	mask_20_inv[mask_20_inv==0 ] = 1
	mask_20_inv[mask_20_inv==-1] = 0
	print(mask_20_inv)
	print(numpy.unique(new_numpy))
	numpy10x_mod = numpy.multiply(mask_20_inv,numpy10x).astype(int)
	print(np.unique(numpy10x_mod))
	numpy10x_mod_mask_6 = (numpy10x_mod==6).astype(int)
	numpy10x_mod_mask_7 = (numpy10x_mod==7).astype(int)
	new_numpy = new_numpy - numpy.multiply(numpy10x_mod_mask_6,new_numpy) - numpy.multiply(numpy10x_mod_mask_7,new_numpy)+numpy.multiply(numpy10x_mod_mask_6,numpy10x_mod)+numpy.multiply(numpy10x_mod_mask_7,numpy10x_mod)
	new_numpy = new_numpy.astype(int)
	print(numpy.unique(new_numpy))
	mod_numpy = copy.deepcopy(new_numpy)
	mod_numpy[mod_numpy == 8] = 0
	mod_numpy[mod_numpy>0] = 255
	print(numpy.unique(mod_numpy))
	img = Image.fromarray(np.uint8(mod_numpy) , 'L')
	print("check image")
	img.save("temp.png")
	#print(img.shape())
	img=cv2.imread('temp.png')
	#print(img.shape
	#print(img.max)
	#print(img)
	#(T, img) = cv2.threshold(mod_numpy, 0.5, 255, 0i)
	img = mod_numpy
	#img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
	img = img.astype('uint8')
	print(img.shape)
	print(np.max(img))
	print(cv2.__version__)
	contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	print("check if val changes")
	for cnt in contours:
		area = cv2.contourArea(cnt)
		if area < remove_small_thre:
			#print("here")
			#print(cnt)
			cv2.fillConvexPoly(img,cnt,0)
			#cv2.fillPoly(img, [cnt], 255)
	print("now")
#	break
	print("check noe")
	print(np.max(img))
	#cv2.imwrite("temp2.png",img)
	img[img==0]=3
	img[img==255] = 0
	img[img==3] = 1
	print("final mask")
	print(np.unique(img))
	temp_img = copy.deepcopy(new_numpy)
	cv2.imwrite("temp_org.png",new_numpy*255)
	print("first numpy vals")
	print(np.unique(new_numpy))
	final_numpy = new_numpy - np.multiply(img,temp_img) + np.multiply(img,np.full(img.shape, 8, dtype=int))
	final_numpy = final_numpy.astype(int)
	print("last numpy vals")
	print(np.unique(final_numpy))
	cv2.imwrite("temp_fin.png",final_numpy*255)
	outputname = i.split('/')[-1]
	print('outputname2',outputname)
	#np.save(out_path+outputname, final_numpy)
	cv2.imwrite(out_path+outputname,final_numpy)




