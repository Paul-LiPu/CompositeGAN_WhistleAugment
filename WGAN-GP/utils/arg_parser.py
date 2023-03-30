# -*- coding: utf-8 -*-
# @Time    : 9/17/20 7:22 PM
# @Author  : Pu Li
# @Email   : pli5270@sdsu.edu
# @File    : arg_parser.py
# @Description:
import argparse

def str2bool(s):
    return s.lower() == 'true'

class ArgParser():
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--exp_name', type=str, required=True, help='Experiment name')
        parser.add_argument('--train_file', type=str, required=True, help='Train data file')
        parser.add_argument('--mode', type=str, required=True, help='pos or neg')
        # parser.add_argument('--test_file', type=str, required=True, help='Test data file')
        parser.add_argument('--log_dir', type=str, default='./logs/', help='test result directory')
        parser.add_argument('--model_dir', type=str, default='./models/', help='model file save directory')
        parser.add_argument('--sample_size', type=int, default=64, help='size of generated sample size')
        parser.add_argument('--model_dim', type=int, default=64, help='dimensionality of the hidden layers')
        parser.add_argument('--batch_size', type=int, default=64, help='batch size')
        parser.add_argument('--critic_iter', type=int, default=5, help='critic iterations')
        parser.add_argument('--gp_lambda', type=float, default=10, help='Gradient penalty lambda hyperparameter')
        parser.add_argument('--train_iters', type=int, default=100000, help='number of iterations for training')
        parser.add_argument('--niter_save_model', type=int, default=10000, help='number of iterations for saving model')

        self.parser = parser

    def parse(self):
        return self.parser.parse_args()