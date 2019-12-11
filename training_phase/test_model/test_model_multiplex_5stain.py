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
import rtree
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


savefolder='/mnt/blobfuse/train-output/ByMZ/multiplex_layer10_softmax/'
if not os.path.exists(savefolder):
    os.makedirs(savefolder)
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
    
def run_model_on_tile(model, naip_tile, inpt_size, batch_size, out_put_times):
    down_weight_padding = 70
    #down_weight_padding = 0
    height = naip_tile.shape[1]
    width = naip_tile.shape[2]

    stride_x = inpt_size - down_weight_padding*2
    stride_y = inpt_size - down_weight_padding*2

    output = np.zeros((height, width, 10), dtype=np.float32)
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
                print('batch.shape',batch.shape)
                print('np.max(batch[0,0:3,...])',np.max(batch[0,0:3,...]))
                model_output = model.eval(np.array(batch))
                print('model_output.shape',model_output.shape)
                print('np.max(model_output[0,0,...])',np.max(model_output[0,0,...]),np.mean(model_output[0,0,...]))
                print('np.max(model_output[0,1,...])',np.max(model_output[0,1,...]),np.mean(model_output[0,1,...]))
                print('np.max(model_output[0,2,...])',np.max(model_output[0,2,...]),np.mean(model_output[0,2,...]))
                
                model_output = np.swapaxes(model_output,1,3)
                model_output = np.swapaxes(model_output,1,2)
               
                model_output = postprocess_output(model_output)
                model_output=expand_output_mao(model_output,out_put_times)
                print('np.max(model_output[0,0,...]) after process',np.max(model_output[0,...,0]),np.mean(model_output[0,...,0]))
                print('np.max(model_output[0,1,...]) after process',np.max(model_output[0,...,1]),np.mean(model_output[0,...,1]))
                print('np.max(model_output[0,2,...]) after process',np.max(model_output[0,...,2]),np.mean(model_output[0,...,2]))
                
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
        model_output = postprocess_output(model_output)
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

    return parser.parse_args(arg_list)

def main():
    program_name = "Model inference script"
    args = do_args(sys.argv[1:], program_name)

    input_fn = args.input_fn
    output_base = args.output_base
    model_fn = args.model_fn
    paral_code = args.paral_code
    paral_max = args.paral_max
    out_put_times=args.out_put_times
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
    print(model_fn)
    model = cntk.load_model(model_fn)

    output_shape = model.outputs[0].shape
    input_shape = model.arguments[0].shape
    print('output_shape',output_shape)
    #assert (input_shape[1]-output_shape[1])/2 == (input_shape[2]-output_shape[2])/2
    #assert int((input_shape[1]-output_shape[1])/2) == (input_shape[1]-output_shape[1])/2
    model_input_size = input_shape[1]

    labels = []
    preds = []
    for i in range(len(pair_list)):
        if (i % paral_max) != paral_code:
            continue
        tic = float(time.time())
        naip_fn = pair_list[i][0]
        label = int(pair_list[i][1])

        #naip_tile = np.array(Image.open(naip_fn).convert('RGB'))
        
        naip_tile=np.load(naip_fn)[0,0:3,...]
        #naip_tile=np.transpose(naip_tile)
        naip_tile=np.squeeze(naip_tile)
        print('lllllllllllllllllllllll',naip_tile.shape)
        edges_overlay=np.array((copy.deepcopy(naip_tile)*255).astype('uint8'))
        edges_overlay=np.swapaxes(edges_overlay,2,0)
        edges_overlay=np.swapaxes(edges_overlay,0,1)
        
        cv2.imwrite(savefolder+naip_fn.split('/')[-1][0:-4]+'_ori.png',edges_overlay)
        print('naip_tile.shape() before',naip_tile.shape)
       
        #naip_tile = np.swapaxes(naip_tile, 1, 2)
        #naip_tile = np.swapaxes(naip_tile, 0, 1)
        print('naip_tile.shape() after',naip_tile.shape)
        print('np.max(naip_tile)',np.max(naip_tile))
        
        output = run_model_on_tile(model, naip_tile.astype(np.float32), model_input_size, 32,out_put_times)
        print('output.shape()',output.shape)
        print('np.max(output)',np.max(output))
        print('np.max(output[...,0])',np.sum(output[...,0]))
        print('np.max(output[...,1])',np.sum(output[...,1]))
        #print('np.max(output[...,2])',np.sum(output[...,2]))
        print('np.min(output)',np.min(output))
        
        #shutil.copyfile(naip_fn,savefolder+naip_fn.split('/')[-1][0:-4]+'_ori.png')
        stain_num=5
        for stain_i in range(stain_num):
            output_classes = output[..., stain_i*2]
            
            w, h = output_classes.shape
            pred = output_classes[w//2, h//2]
            print('{} {} {}'.format(naip_fn, label, pred))

            labels.append(label)
            preds.append(pred)
            print('max min',np.max(output_classes),np.min(output_classes))
            output_classes=(output_classes*255).astype('uint8')
            cv2.imwrite(savefolder+os.path.basename(naip_fn)[0:-4]+'_'+str(stain_i)+'.png',output_classes)
            #output_classes=Image.fromarray(output_classes.transpose())
            #output_classes=Image.fromarray(output_classes)
            #output_classes.save(savefolder+os.path.basename(naip_fn)[0:-4]+'_'+str(stain_i)+'.png')
            print(naip_fn)
            print(naip_fn.split('/'))
            
            
            
            heatmap1,thre= cv2.threshold(output_classes,20,255,cv2.THRESH_BINARY)
            thre_edge=edge(thre,2)
            edges_overlay=np.clip(edges_overlay+thre_edge,0,255)
            #heatmap=np.zeros_like(ori_image)
            #open_cv_image = np.array(output_classes) 
            # Convert RGB to BGR 
            #open_cv_image = open_cv_image[:, :, ::-1].copy() 
            #heatmap[:,:,1]=output_classes
            #overlay=np.clip(ori_image+(heatmap)*0.5,0,255)
        cv2.imwrite(savefolder+naip_fn.split('/')[-1][0:-4]+'_overlay.png',edges_overlay)
        
    #fpr, tpr, thresholds = metrics.roc_curve(labels, preds, pos_label=1)
    #print('Final AUC: {}'.format(metrics.auc(fpr, tpr)))

if __name__ == "__main__":
    main()

