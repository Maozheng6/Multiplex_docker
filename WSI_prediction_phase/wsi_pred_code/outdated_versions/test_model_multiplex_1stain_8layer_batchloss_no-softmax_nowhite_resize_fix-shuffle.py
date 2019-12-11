#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# pylint: skip-file
#
# Caleb Robinson <calebrob6@gmail.com>
# and
# Le Hou <lehou0312@gmail.com>
"""Script for running a saved model file on a list of NAIP+Landsat images.
"""
# Stdlib imports
import sys
import os
import time
import shutil
import datetime
import argparse
import logging
import subprocess
import tempfile
import pickle
import random
#import rtree
import math
import cv2
import copy
# Library imports
import numpy as np
import pandas as pd

import cntk

import rasterio
import rasterio.mask
import fiona
import fiona.transform
import shapely
import shapely.geometry
from scipy import misc
from PIL import Image
from sklearn import metrics
import matplotlib.pyplot as plt


# Setup
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger("rasterio._base").setLevel(logging.WARNING)
logging.getLogger("rasterio._io").setLevel(logging.WARNING)
logging.getLogger("shapely.geos").setLevel(logging.WARNING)
logging.getLogger("Fiona").setLevel(logging.WARNING)

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def edge(img,dim):
    img2=np.zeros((img.shape[0],img.shape[1],3)).astype('float')
    img1=cv2.Canny(img,100,200)
    kernel = np.ones((3,3),np.uint8)
    img1=cv2.dilate(img1,kernel,iterations = 1)

    for i in range(img2.shape[2]):
        if dim==12:
            img2[:,:,1]=img1.astype('float')
            img2[:,:,2]=img1.astype('float')
        else:
            if i==dim:
                img2[:,:,i]=img1.astype('float')
            else:
                img2[:,:,i]=-img1.astype('float')

    '''
    for i in img1.shape[2]:
        if i!=dim:
            img1[:,:,i]=0
    '''
    return img2
def postprocess_output(output):
    output_max = np.max(output, axis=3, keepdims=True)
    #[?] why -output_max
    exps = np.exp(output-output_max)
    exp_sums = np.sum(exps, axis=3, keepdims=True)
    return exps/exp_sums

def postprocess_output_2(output):
    for i in range(5):
        output_temp=output[:,:,:,i*2:i*2+2]
        output_max = np.max(output_temp, axis=3, keepdims=True)
        #[?] why -output_max
        exps = np.exp(output_temp-output_max)
        exp_sums = np.sum(exps, axis=3, keepdims=True)
        output[:,:,:,i*2:i*2+2]=exps/exp_sums
    return output
def expand_output_mao(output,times):
    row=output.shape[1]
    col=output.shape[2]
    print(output.shape)
    new_output=np.ones((output.shape[0],output.shape[1]*times,output.shape[2]*times,output.shape[3]))
    for i in range(row):
        for j in range(col):
            for k in range(output.shape[0]):
                for l in range(output.shape[3]):
                    new_output[k,i*times:(i+1)*times,j*times:(j+1)*times,l]=output[k,i,j,l]
    return new_output

def run_model_on_tile(model, naip_tile, inpt_size, batch_size, out_put_times,stain_num):
    down_weight_padding = 70
    #down_weight_padding = 0
    height = naip_tile.shape[1]
    width = naip_tile.shape[2]

    stride_x = inpt_size - down_weight_padding*2
    stride_y = inpt_size - down_weight_padding*2

    output = np.zeros((height, width, stain_num+1), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32) + 0.000000001
    kernel = np.ones((inpt_size, inpt_size), dtype=np.float32) * 0.1
    #Mao kernel[10:-10, 10:-10] = 1
    kernel[20:-20, 20:-20] = 1
    kernel[down_weight_padding:down_weight_padding+stride_y,
           down_weight_padding:down_weight_padding+stride_x] = 5

    batch = []
    batch_indices = []
    batch_count = 0

    for y_index in (list(range(0, height - inpt_size, stride_y)) + [height - inpt_size,]):
        for x_index in (list(range(0, width - inpt_size, stride_x)) + [width - inpt_size,]):
            naip_im = naip_tile[:, y_index:y_index+inpt_size, x_index:x_index+inpt_size]

            batch.append(naip_im)
            batch_indices.append((y_index, x_index))
            batch_count+=1

            ########################
            # run batch
            if batch_count >= batch_size:
                batch=np.array(batch)
                model_output = model.eval(np.array(batch))
                model_output = np.swapaxes(model_output,1,3)
                model_output = np.swapaxes(model_output,1,2)
                model_output=expand_output_mao(model_output,out_put_times)

                for i, (y, x) in enumerate(batch_indices):
                    output[y:y+inpt_size, x:x+inpt_size] += model_output[i] * kernel[..., np.newaxis]
                    counts[y:y+inpt_size, x:x+inpt_size] += kernel
                batch = []
                batch_indices = []
                batch_count = 0
            # run batch
            ########################

    if batch_count > 0:
        model_output = model.eval(np.array(batch))

        model_output = np.swapaxes(model_output,1,3)
        model_output = np.swapaxes(model_output,1,2)
        model_output=expand_output_mao(model_output,out_put_times)
        for i, (y, x) in enumerate(batch_indices):
            output[y:y+inpt_size, x:x+inpt_size] += model_output[i] * kernel[..., np.newaxis]
            counts[y:y+inpt_size, x:x+inpt_size] += kernel

    return output / counts[..., np.newaxis]


#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------

def do_args(arg_list, name):
    parser = argparse.ArgumentParser(description=name)

    parser.add_argument("-i", "--input", action="store", dest="input_fn", type=str, required=True, \
        help="Input file name (pair list CSV file with 'patch_path' and 'label' columns)"
    )
    parser.add_argument("-o", "--output", action="store", dest="output_base", type=str, required=True, \
        help="Output directory to store predictions"
    )
    parser.add_argument("-m", "--model", action="store", dest="model_fn", type=str, required=True, \
        help="Path to CNTK .model file to use"
    )

    parser.add_argument("--paral_code", action="store", dest="paral_code", type=int, default=0, \
        help="Used for parallelization"
    )
    parser.add_argument("--paral_max", action="store", dest="paral_max", type=int, default=1, \
        help="Used for parallelization"
    )
    parser.add_argument("--gpuid", action="store", dest="gpuid", type=int, required=False, default=-1,
        help="GPU ID for cntk"
    )
    parser.add_argument("--out_put_times", action="store", dest="out_put_times", type=int, required=False, default=1,
        help="out_put_times"
    )
    parser.add_argument("--stain_num", action="store", dest="stain_num", type=int, required=False, default=1,
        help="stain_num"
    )
    parser.add_argument("--channel_times", action="store", dest="channel_times",       type=int, required=False, default=1,
                    help="channel_times"
                        )

    parser.add_argument("--resize_ratio", action="store", dest="resize_ratio", type=float, required=False, default=1,  help="resize_ratio")

    return parser.parse_args(arg_list)

def main():
    program_name = "Model inference script"
    args = do_args(sys.argv[1:], program_name)

    input_fn = args.input_fn
    output_base = args.output_base
    savefolder=output_base
    if not os.path.exists(savefolder):
        os.makedirs(savefolder)
    model_fn = args.model_fn
    paral_code = args.paral_code
    paral_max = args.paral_max
    out_put_times=args.out_put_times
    stain_num=args.stain_num
    channel_times=args.channel_times
    resize_ratio=args.resize_ratio
    if args.gpuid >= 0:
        cntk.device.try_set_default_device(cntk.device.gpu(args.gpuid))

    logging.info("Starting %s at %s" % (program_name, str(datetime.datetime.now())))
    start_time = float(time.time())

    # Make sure the input files are OK and ensure output directory exist
    if not os.path.isfile(input_fn):
        logging.error("Input file does not exist, exiting")
        return

    if not os.path.isfile(model_fn):
        logging.error("Model file does not exist, exiting")
        return

    try:
        input_df = pd.read_csv(input_fn)
        pair_list = input_df[["patch_path", "label"]].values
    except Exception as e:
        logging.error("Could not load the input file")
        logging.error(e)
        return

    if not os.path.exists(output_base):
        logging.error("Output directory does not exist, making output dirs: %s" % (output_base))
        os.makedirs(output_base)
    model = cntk.load_model(model_fn)

    output_shape = model.outputs[0].shape
    input_shape = model.arguments[0].shape
    model_input_size = input_shape[1]

    #labels = []
    preds = []

    name_list = [xx[0] for xx in pair_list]
    #label_list = [xx[1] for xx in pair_list]
    random.shuffle(name_list)
    print('total files',len(name_list))
    for i in range(len(name_list)):
        #if (i % paral_max) != paral_code:
        #    print('(i % paral_max) != paral_code',i,para_max,paral_code)
        #    continue
        tic = float(time.time())
        naip_fn = name_list[i]
        #label = int(pair_list[i][1])
        if os.path.isfile(savefolder+os.path.basename(naip_fn)[0:-4]+'_8fig.png'):
            print('File exits, skip this file:',savefolder+os.path.basename(naip_fn)[0:-4]+'_8fig.png')
            continue
        else:
            print('----------------------------compute this file:',savefolder+os.path.basename(naip_fn)[0:-4]+'_8fig.png')
        print(naip_fn)
        naip_tile = cv2.imread(naip_fn)
        #################################
        #resize to correct size
        print('resizing input')
        naip_tile = cv2.resize(naip_tile,(int(naip_tile.shape[1]*resize_ratio),int(naip_tile.shape[0]*resize_ratio)))
        print('input size after resize',naip_tile.shape)
        ##############################
        #skip this image if it's white image
        white_image=False
        if naip_tile.std()<6 and naip_tile.mean()>215:
            white_image=True
            print('White_image!!!')
        naip_tile=naip_tile/255.0
        naip_tile=naip_tile[:,:,::-1]
        for che in range(channel_times):
            if che == 0:
                naip_im_channels = naip_tile
            else:
                naip_im_channels = np.concatenate([naip_im_channels,naip_tile],axis=2)
        naip_tile=naip_im_channels
        naip_tile=np.swapaxes(naip_tile,2,0)
        naip_tile=np.swapaxes(naip_tile,2,1)
        naip_tile=np.squeeze(naip_tile)
        edges_overlay=np.array((copy.deepcopy(naip_tile)*255).astype('uint8'))
        edges_overlay=np.swapaxes(edges_overlay,2,0)
        edges_overlay=np.swapaxes(edges_overlay,0,1)
        edges_overlay=edges_overlay[:,:,0:3]
        ori_img=edges_overlay
        if white_image!=True:
            output = run_model_on_tile(model, naip_tile.astype(np.float32), model_input_size, 32,out_put_times,stain_num)
        else:
            height = naip_tile.shape[1]
            width = naip_tile.shape[2]
            output =  np.zeros((height,width, stain_num+1), dtype=np.float32)
        print('output.shape',output.shape)
        stain_pred=[]
        overlayed=[]
        for stain_i in range(0,stain_num+1):
            output_classes = output[..., stain_i]

            w, h = output_classes.shape
            pred = output_classes[w//2, h//2]
            #labels.append(label)
            preds.append(pred)
            output_classes=(output_classes*255).astype('uint8')
            stain_pred.append(output_classes)

            print('save heatmap')
            cv2.imwrite(savefolder+naip_fn.split('/')[-1][0:-4]+'_'+str(stain_i)+'.png',cv2.resize(output_classes,(int(output_classes.shape[1]/resize_ratio),int(output_classes.shape[0]/resize_ratio))))
            heatmap1,thre= cv2.threshold(output_classes,210,255,cv2.THRESH_BINARY)
            kernel = np.ones((3,3),np.uint8)
            thre=cv2.erode(thre,kernel,iterations = 1)
            thre=cv2.dilate(thre,kernel,iterations = 1)
            thre_edge=edge(thre,1)
            edges_overlay0=np.clip(edges_overlay+thre_edge,0,255).astype('uint8')
            edges_overlay1=cv2.resize(edges_overlay0,(int(edges_overlay0.shape[1]/4),int(edges_overlay0.shape[0]/4)))
            cv2.imwrite(savefolder+naip_fn.split('/')[-1][0:-4]+'_'+str(stain_i)+'_overlay.png',edges_overlay1)
            overlayed.append(edges_overlay1)
        #cv2.imwrite(savefolder+naip_fn.split('/')[-1][0:-4]+'_overlay.png',edges_overlay)

        per=4
        f, axarr = plt.subplots(2, per, figsize=( 15, 12))

        stain_colors=['CD16: black','CD 20: pink','CD3: yellow','CD4: cyan','CD8: purple','K17 +','K17 -','BG']
        for idx in range(1+stain_num):
            if idx==0:
                axarr[idx//per][idx%per].imshow(ori_img[:,:,::-1]/255)
                axarr[idx//per][idx%per].set_title('RGB')
                axarr[idx//per][idx%per].axis('off')


            else:
                axarr[idx//per][idx%per].imshow(overlayed[idx-1][:,:,::-1]/255)
                axarr[idx//per][idx%per].set_title('"%s"' % (stain_colors[idx-1]))
                axarr[idx//per][idx%per].axis('off')
        f.savefig(savefolder+os.path.basename(naip_fn)[0:-4]+'_8fig.png', bbox_inches='tight', pad_inches=0)
        plt.close(f)
        if os.path.isfile(savefolder+os.path.basename(naip_fn)[0:-4]+'_8fig.png'):
            print('!!!!!!!!!!!!!!!saved:',savefolder+os.path.basename(naip_fn)[0:-4]+'_8fig.png')
        else:
            print('FFFFFFFFFFFFFFFail saved:',savefolder+os.path.basename(naip_fn)[0:-4]+'_8fig.png')
    print('This slide is Done, i=',i)

if __name__ == "__main__":
    main()

