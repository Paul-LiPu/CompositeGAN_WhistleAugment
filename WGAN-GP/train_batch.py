# -*- coding: utf-8 -*-
# @Time    : 9/17/20 7:23 PM
# @Author  : Pu Li
# @Email   : pli5270@sdsu.edu
# @File    : train_batch.py
# @Description:

# import library
import os, sys
sys.path.append(os.getcwd())
from utils import util
import re
from utils.m_dataset import read_h5_length
from logger import getLogger
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_meta_dir', type=str, required=True, help='Directory containing the meta files for positive and negative files')

config = parser.parse_args()

data_meta_dir = config.data_meta_dir
batch_name = os.path.basename(data_meta_dir)

neg_data_files = glob.glob(os.path.join(data_meta_dir, 'neg*'))
pos_data_files = glob.glob(os.path.join(data_meta_dir, 'pos*'))

# neg_train_iters = 30000
neg_train_iters = 50000
neg_train_save = 1000
neg_model_dir = os.path.join('./models', batch_name)
neg_log_dir = os.path.join('./logs', batch_name)
# pos_train_iters = 30000
pos_train_iters = 50000
pos_train_save = 1000
pos_model_dir = os.path.join('./models', batch_name)
pos_log_dir = os.path.join('./logs', batch_name)
util.checkDirs([neg_model_dir, neg_log_dir, pos_model_dir, pos_log_dir])

import time
timestamp = time.strftime("%Y%m%d_%H-%M-%S", time.localtime())
logFile = os.path.join('./logs/', 'train_batch_' + timestamp + '.log')
txt_logger = getLogger(logFile)

# Run negative experiment.
pattern = '(.*).txt'
matcher = re.compile(pattern)
for txt_file in neg_data_files:
    filename = os.path.basename(txt_file)
    results = matcher.findall(filename)
    exp_name = results[0]
    run_code = 'python train.py --exp_name ' + exp_name + ' --mode neg --train_file ' + txt_file + \
        ' --train_iters ' + str(neg_train_iters) + ' --niter_save_model ' + str(neg_train_save) + \
        ' --model_dir ' + neg_model_dir + ' --log_dir ' + neg_log_dir
    txt_logger.info(run_code)
    os.system(run_code)


# Run positive experiment.
pattern = '(.*).txt'
matcher = re.compile(pattern)
for txt_file in pos_data_files:
    filename = os.path.basename(txt_file)
    results = matcher.findall(filename)
    exp_name = results[0]
    run_code = 'python train.py --exp_name ' + exp_name + ' --mode pos --train_file ' + txt_file + \
        ' --train_iters ' + str(pos_train_iters) + ' --niter_save_model ' + str(pos_train_save) + \
        ' --model_dir ' + pos_model_dir + ' --log_dir ' + pos_log_dir
    txt_logger.info(run_code)
    os.system(run_code)
