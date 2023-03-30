# -*- coding: utf-8 -*-
# @Time    : 9/17/20 7:23 PM
# @Author  : Pu Li
# @Email   : pli5270@sdsu.edu
# @File    : train_batch.py
# @Description:

# import library
import os, sys
sys.path.append('../')
from util import util_func as util
from cycleGAN import utils
import re
from data.m_dataset import read_h5_length, HDF5_Dataset, HDF5_Dataset_in_RAM
from util.logger import getLogger
import numpy as np
import torch
import math
import h5py
from util.save_images import save_images

import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('--pos_data_dir', type=str, required=True, help='Path to positive wgan model')
# parser.add_argument('--neg_data_dir', type=str, required=True, help='number of patches to generate')
# parser.add_argument('--cycle_name', type=str, required=True, help='number of patches to generate')
# parser.add_argument('--cycle_which_epoch', type=int, required=True, help='number of patches to generate')
# parser.add_argument('--cycle_checkpoints_dir', type=str, default='./checkpoints', help='number of patches to generate')
# parser.add_argument('--cycle_net_type', type=str, default='whistle_cycle_deepContour_negID_residual', help='number of patches to generate')
# parser.add_argument('--output_dir', type=str, required=True, help='directory to save data')
# config = parser.parse_args()


cycle_gan, config = utils.create_cyclegan2()
cycle_gan.load_networks(config.which_epoch)
pos_output_dir = config.output_dir
util.checkDir(pos_output_dir)

pos_data_file = '%s/data.h5' % (config.pos_data_dir)
neg_data_file = '%s/data.h5' % (config.neg_data_dir)

pos_name = os.path.basename(os.path.dirname(pos_data_file))
neg_name = os.path.basename(os.path.dirname(neg_data_file))

# cycle_which_epoch = config.cycle_which_epoch
# cycle_name = config.cycle_name
# cycle_gan = utils.create_cyclegan(config.cycle_net_type, config.cycle_checkpoints_dir, cycle_name)

batch_size = 4
pos_dataset = HDF5_Dataset([pos_data_file], batch_size)
neg_dataset = HDF5_Dataset([neg_data_file], batch_size)
num_data = len(pos_dataset)
assert len(pos_dataset) == len(neg_dataset)
data = []
label = []

pos_data, pos_label = next(pos_dataset)
# print(pos_label.shape)
label.append(pos_label)
neg_data, neg_label = next(neg_dataset)
# print(neg_data.shape)
pos_label = torch.from_numpy(np.asarray(pos_label)).cuda()
neg_data = torch.from_numpy(np.asarray(neg_data)).cuda()
pos_label = utils.shift_value(pos_label, True)
neg_data = utils.shift_value(neg_data, True)
fake_pos_res = utils.generate_fakeB_residual(cycle_gan, pos_label, neg_data)
# fake_pos_res = utils.shift_value(fake_pos_res)
fake_pos_res /= 2
fake_res = fake_pos_res.clone().cpu().detach().numpy()
fake_pos_res += utils.shift_value(neg_data)
data.append(fake_pos_res.cpu().detach().numpy())

data = np.concatenate(data, axis=0)
label = np.concatenate(label, axis=0)
save_images(data, os.path.join(config.output_dir, 'epoch%03d_fake_B.png' % (int(config.which_epoch))))
save_images(fake_res, os.path.join(config.output_dir, 'epoch%03d_fake_B_res.png' % (int(config.which_epoch))))
save_images(label, os.path.join(config.output_dir, 'epoch%03d_real_A_front.png' % (int(config.which_epoch))))
