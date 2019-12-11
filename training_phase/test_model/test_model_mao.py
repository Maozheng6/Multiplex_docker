#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# pylint: skip-file
#
# Caleb Robinson <calebrob6@gmail.com>
# and
# Le Hou <lehou0312@gmail.com>
"""Script for training Kolya's U-NET variant model.
"""
# Stdlib imports
import sys
import os
import time
import datetime
import argparse
import logging
import string
import warnings
warnings.filterwarnings("ignore")
# Library imports
import numpy as np
import pandas as pd

import cntk
import rasterio

# Custom imports
import ModelLib
from DataHandle import get_nlcd_stats
from MyDataSources import MyDataSource
from PIL import Image
cid, nlcd_dist, nlcd_var = get_nlcd_stats()

# Setup
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def softmax(output):
    output_max = np.max(output, axis=-1, keepdims=True)
    exps = np.exp(output-output_max)
    exp_sums = np.sum(exps, axis=-1, keepdims=True)
    return exps/exp_sums

def do_args(arg_list, name):
    parser = argparse.ArgumentParser(description=name)

    parser.add_argument("-v", "--verbose", action="store_true", default=False,
        help="Enable verbose debugging"
    )
    parser.add_argument("--name", action="store", dest="exp_name", type=str, default="_random",
        help="Name of experiment (for logging purposes)"
    )
    parser.add_argument("--gpuid", action="store", dest="gpuid", type=int, required=False, default=-1,
        help="GPU ID for cntk"
    )
    parser.add_argument("--train_patch_list", action="store", dest="train_patch_list", type=str,
        default="/mnt/blobfuse/cnn-minibatches/v1_list.txt",
        help="List of training patches"
    )
    parser.add_argument("--test_patch_list", action="store", dest="test_patch_list", type=str,
        default="/mnt/blobfuse/cnn-minibatches/v1_small_list.txt",
        help="List of testing patches"
    )
    parser.add_argument("-o", "--output", action="store", dest="output_base", type=str,
        default="/mnt/blobfuse/model-output/",
        help="Directory for storing model run output"
    )

    parser.add_argument("--train_label_dis", action="store", dest="train_label_dis", type=str,
        default="0_10_1",
        help="How many train_label_dis patches per minibatch"
    )
    parser.add_argument("--superres", action="store", dest="superres", type=float, required=False, default=1,
        help="Supre-res loss weight"
    )
    parser.add_argument("--highres", action="store", dest="highres", type=float, required=False, default=30,
        help="High-res loss weight"
    )

    return parser.parse_args(arg_list)

def main():
    program_name = "Model training script"
    args = do_args(sys.argv[1:], program_name)

    verbose = args.verbose
    exp_name = args.exp_name
    output_base = args.output_base
    train_patch_list = args.train_patch_list
    test_patch_list = args.test_patch_list

    if args.gpuid >= 0:
        cntk.device.try_set_default_device(cntk.device.gpu(args.gpuid))
    train_label_dis = [float(x) for x in args.train_label_dis.split('_')]
    dis_sum = sum(train_label_dis)
    train_label_dis = [x/dis_sum for x in train_label_dis]
    test_label_dis = [1.0 for x in train_label_dis]
    superres = float(args.superres)
    highres = float(args.highres)

    logging.info("Starting %s at %s" % (program_name, str(datetime.datetime.now())))
    start_time = float(time.time())


    if not os.path.exists(output_base):
        logging.error("Output directory does not exist, making output dirs: %s" % (output_base))
        os.makedirs(output_base)

    if exp_name == "_random":
        exp_name = ''.join(np.random.choice(list(string.ascii_letters), 10, replace=True))
    else:
        if os.path.exists(os.path.join(output_base, exp_name, "notes.txt")):
            logging.warning("Experiment with the same name has already been run!")

    exp_output_base = os.path.join(output_base, exp_name)
    if not os.path.exists(exp_output_base):
        os.makedirs(exp_output_base)

    logfile_base = os.path.join(output_base, "logs/")
    if not os.path.exists(logfile_base):
        os.makedirs(logfile_base)

    # Start of program logic
    logging.info("Running experiment %s" % (exp_name))
    logging.info("Important settings: {} {} {}".format(
        train_label_dis, superres, highres))


    #----------------------------------------------------------------
    # Setup training arguments (by hand for now)
    #----------------------------------------------------------------

    exp_description = 'Just doing random experiments'

    f = open(os.path.join(exp_output_base, "notes.txt"), "w")
    f.write(exp_name+"\n")
    f.write(exp_description)
    f.close()

    mb_size = 20
    epoch_size = 200
    num_test_minibatches = 2
    max_epochs = 90
    edge_sigma = 3
    edge_loss_boost = 30.0

    super_res_loss_weight = superres
    high_res_loss_weight = highres

    # Weighting the loss of each NLCD class
    super_res_class_weight = [0.0,] + [1.0,]*10

    lr_adjustment_factor = 1.0
    num_stack_layers = 3 # A parameter for defining the model architecture

    num_nlcd_classes, num_landcover_classes = nlcd_dist.shape
    num_color_channels = 3
    block_size = 160 # ROIs will be 240 pixels x 240 pixels

    # Create the minibatch source
    f_dim = (num_color_channels, block_size, block_size)
    l_dim = (num_landcover_classes, block_size, block_size)
    m_dim = (num_nlcd_classes, block_size, block_size)
    c_dim = (num_nlcd_classes, num_landcover_classes) # same dims for interval centers and radii

    training_minibatch_source = MyDataSource(
        f_dim=f_dim,
        l_dim=l_dim,
        m_dim=m_dim,
        c_dim=c_dim,
        highres_only=(superres<0.001),
        edge_sigma=edge_sigma,
        edge_loss_boost=edge_loss_boost,
        patch_list=train_patch_list,
    )
    testing_minibatch_source = MyDataSource(
        f_dim=f_dim,
        l_dim=l_dim,
        m_dim=m_dim,
        c_dim=c_dim,
        highres_only=(superres<0.001),
        edge_sigma=edge_sigma,
        edge_loss_boost=edge_loss_boost,
        patch_list=test_patch_list,
    )

    input_im, lc, lc_weight_map, mask, interval_center, interval_radius, output_tensor, high_res_loss, loss = \
        ModelLib.get_model(
        f_dim, c_dim, l_dim, m_dim, num_stack_layers,
        super_res_class_weight, super_res_loss_weight, high_res_loss_weight)

    training_input_map = {
        input_im: training_minibatch_source.streams.features,
        lc: training_minibatch_source.streams.landcover,
        lc_weight_map: training_minibatch_source.streams.lc_weight_map,
        mask: training_minibatch_source.streams.masks,
        interval_center: training_minibatch_source.streams.interval_centers,
        interval_radius: training_minibatch_source.streams.interval_radii
    }
    testing_input_map = {
        input_im: testing_minibatch_source.streams.features,
        lc: testing_minibatch_source.streams.landcover,
        lc_weight_map: testing_minibatch_source.streams.lc_weight_map,
        mask: testing_minibatch_source.streams.masks,
        interval_center: testing_minibatch_source.streams.interval_centers,
        interval_radius: testing_minibatch_source.streams.interval_radii
    }

    # Start training
    trainer, tensorboard = ModelLib.make_trainer(
        epoch_size, mb_size, output_tensor, high_res_loss, loss, max_epochs, 0,
        1, lr_adjustment_factor, os.path.join(logfile_base, exp_name)
    )

    cntk.logging.log_number_of_parameters(output_tensor)

    logging.info("Epoch size: {} minibatch iterations. Minibatch size: {}.".format(epoch_size, mb_size))

    mb_num = 0
    epoch_num = 0
    while mb_num < epoch_size * max_epochs:
        train_minibatch_data = training_minibatch_source.next_minibatch(
                mb_size, train_label_dis)

        trainer.train_minibatch({
            k: train_minibatch_data[v]
            for k,v in training_input_map.items()
        })
        mb_num += 1

        if mb_num % epoch_size == 0:
        #if mb_num % 20 == 0:
            epoch_num += 1
            trainer.summarize_training_progress()
            trainer.model.save(os.path.join(exp_output_base, "cnn_%d.model" % (epoch_num)))
            #mao
            logging.info("Finished %s epochs, avg time/epoch %0.4f seconds" % (str(epoch_num), (time.time() - start_time)/epoch_num))
            for j in range(num_test_minibatches):
                test_minibatch_data = testing_minibatch_source.next_minibatch(
                        mb_size, test_label_dis)
                trainer.test_minibatch({
                    k: test_minibatch_data[v]
                    for k,v in testing_input_map.items()
                })
            trainer.summarize_test_progress()

            test_img, _, _ = testing_minibatch_source.get_random_instance()
            #test_img=int(test_img)
            test_pred = output_tensor.eval(test_img[np.newaxis, ...])
            test_pred = np.swapaxes(test_pred,1,3)
            test_pred = np.swapaxes(test_pred,1,2)
            test_pred = (softmax(np.squeeze(test_pred))[..., -1]*255.0).astype(np.uint8)

            # Save test prediction tile
            Image.fromarray(test_pred).save(
                os.path.join(exp_output_base, "pred_%d.png" % (epoch_num)))

            # Save test image tile
            test_img = np.swapaxes(test_img, 0, 2)
            test_img = np.swapaxes(test_img, 0, 1)
            test_img_out_file = os.path.join(exp_output_base, "image_%d.png" % (epoch_num))
            Image.fromarray((255*test_img).astype(np.uint8)[:,:,0:3]).save(test_img_out_file)

            tensorboard.flush()

    trainer.model.save(os.path.join(exp_output_base,"exp_%s_final.model" % (exp_name)))

    logging.info("Finished %s in %0.4f seconds" % (program_name, time.time() - start_time))

if __name__ == "__main__":
    main()
