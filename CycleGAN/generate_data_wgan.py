# -*- coding: utf-8 -*-
# @Time    : 9/17/20 7:23 PM
# @Author  : Pu Li
# @Email   : pli5270@sdsu.edu
# @File    : train_batch.py
# @Description:

# import library
import os, sys
sys.path.append('../')
from cycleGAN import utils
from util import util_func
import re
from data.m_dataset import read_h5_length
from util.logger import getLogger
import torch
import math
import numpy as np
import h5py
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--pos_model', type=str, required=True, help='Path to positive wgan model')
parser.add_argument('--num_patch', type=int, required=True, help='number of patches to generate')
parser.add_argument('--output_dir', type=str, required=True, help='directory to save data')

config = parser.parse_args()

pos_model = config.pos_model
output_dir = './generated_data'
util_func.checkDir(output_dir)

# Run negative experiment.
model_name = os.path.basename(os.path.dirname(pos_model))
model_iter = os.path.basename(pos_model).split('iter')[1]

num_patch = config.num_patch
batch_size = 50

pos_label_gan = utils.create_wgan('wgan2')
pos_label_gan.load_state_dict(torch.load(pos_model)['GNET'])


# def thresh(data):
#     data = data.reshape((data.size[0], -1))
#     pos_pixel = torch.sum(data > 0.5, dim=1)
#     entropy = torch.sum(data * torch.log(data + 1e-5), dim=1)
#     # select_idx = torch.logical_and(pos_pixel > 64, entropy > -70)
#     select_idx = (pos_pixel > 64) &  (entropy > -70)
#     return select_idx

def thresh(data):
    data = data.reshape((data.shape[0], -1))
    pos_pixel = np.sum(data > 0.5, axis=1)
    entropy = np.sum(data * np.log(data + 1e-5), axis=1)
    select_idx = np.logical_and(pos_pixel > 64, entropy > -70)
    return select_idx


fake_data = []
count = 0
while (count < num_patch):
    fake_pos_label = utils.generate_image(pos_label_gan, batch_size)
    fake_pos_label = utils.shift_value(fake_pos_label)
    fake_pos_label = fake_pos_label.cpu().detach().numpy()
    # fake_pos_label = fake_pos_label.squeeze()
    good_data = fake_pos_label[thresh(fake_pos_label), ]
    count += good_data.shape[0]
    fake_data.append(good_data)
    print(count)
fake_data = np.concatenate(fake_data, axis=0)
fake_data = fake_data[:num_patch, ]
# pos_output_dir = os.path.join(output_dir, '%s_iter%d_pos_good' % (model_name, model_iter))
pos_output_dir = config.output_dir
util_func.checkDir(pos_output_dir)

hf = h5py.File(pos_output_dir + '/data.h5', 'w')
hf.create_dataset('data', data=fake_data, dtype='f')
hf.create_dataset('label', data=fake_data, dtype='f')
hf.close()







