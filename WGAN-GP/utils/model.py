# -*- coding: utf-8 -*-
# @Time    : 9/17/20 7:25 PM
# @Author  : Pu Li
# @Email   : pli5270@sdsu.edu
# @File    : model.py
# @Description:

import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        DIM = config.model_dim
        preprocess = nn.Sequential(
            nn.Linear(128, 4*4*4*DIM),
            nn.ReLU(True),
        )
        block1 = nn.Sequential(
            nn.Conv2d(4*DIM, 8*DIM, 3, padding=1),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.Conv2d(2*DIM, 4*DIM, 3,padding=1),
            nn.ReLU(True),
        )
        block3 = nn.Sequential(
            nn.Conv2d(DIM, 2*DIM, 3,padding=1),
            nn.ReLU(True),
        )
        conv_out = nn.Conv2d(DIM//2, 4, 3, padding=1)

        self.block1 = block1
        self.block2 = block2
        self.block3 = block3
        self.conv_out = conv_out
        self.preprocess = preprocess
        self.Tanh = nn.Tanh()
        self.DIM = DIM
        self.OUTPUT_DIM = config.sample_size ** 2

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 4*self.DIM, 4, 4) # [batch, 4*DIM, 4, 4]
        output = self.block1(output) # [batch, 8*DIM, 4, 4]
        output = nn.functional.pixel_shuffle(output, 2) # [batch, 2*DIM, 8, 8]
        output = self.block2(output) # [batch, 4*DIM, 8, 8]
        output = nn.functional.pixel_shuffle(output, 2)  # [batch, DIM, 16, 16]
        output = self.block3(output)  # [batch, 2*DIM, 16, 16]
        output = nn.functional.pixel_shuffle(output, 2)  # [batch, DIM/2, 32, 32]
        output = self.conv_out(output) # [batch, 4, 32, 32]
        output = nn.functional.pixel_shuffle(output, 2)  # [batch, 1, 64, 64]
        output = self.Tanh(output)
        return output.view(-1, self.OUTPUT_DIM)


class Generator2(nn.Module):
    def __init__(self, config):
        super(Generator2, self).__init__()
        DIM = config.model_dim
        preprocess = nn.Sequential(
            nn.Linear(128, 4*4*8*DIM),
            nn.ReLU(True),
        )
        block1 = nn.Sequential(
            nn.Conv2d(8*DIM, 16*DIM, 3, padding=1),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.Conv2d(4*DIM, 8*DIM, 3,padding=1),
            nn.ReLU(True),
        )
        block3 = nn.Sequential(
            nn.Conv2d(2*DIM, 4*DIM, 3,padding=1),
            nn.ReLU(True),
        )
        conv_out = nn.Conv2d(DIM, 8, 3, padding=1)

        self.block1 = block1
        self.block2 = block2
        self.block3 = block3
        self.conv_out = conv_out
        self.preprocess = preprocess
        self.Tanh = nn.Tanh()
        self.DIM = DIM
        self.OUTPUT_DIM = config.sample_size ** 2 * 2

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 8*self.DIM, 4, 4) # [batch, 8*DIM, 4, 4]
        output = self.block1(output) # [batch, 16*DIM, 4, 4]
        output = nn.functional.pixel_shuffle(output, 2) # [batch, 4*DIM, 8, 8]
        output = self.block2(output) # [batch, 8*DIM, 8, 8]
        output = nn.functional.pixel_shuffle(output, 2)  # [batch, 2*DIM, 16, 16]
        output = self.block3(output)  # [batch, 4*DIM, 16, 16]
        output = nn.functional.pixel_shuffle(output, 2)  # [batch, DIM, 32, 32]
        output = self.conv_out(output) # [batch, 8, 32, 32]
        output = nn.functional.pixel_shuffle(output, 2)  # [batch, 2, 64, 64]
        output = self.Tanh(output)
        return output

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        DIM = config.model_dim
        main = nn.Sequential(
            nn.Conv2d(1, DIM, 5, stride=2, padding=2), # [batch, DIM, 32, 32]
            nn.ReLU(True),
            nn.Conv2d(DIM, 2*DIM, 5, stride=2, padding=2),# [batch, 2*DIM, 16, 16]
            nn.ReLU(True),
            nn.Conv2d(2*DIM, 4*DIM, 5, stride=2, padding=2),# [batch, 4*DIM, 8, 8]
            nn.ReLU(True),
            nn.Conv2d(4 * DIM, 8*DIM, 5, stride=2, padding=2),  # [batch, 8*DIM, 4, 4]
            nn.ReLU(True),
        )
        self.main = main
        self.output = nn.Linear(4*4*8*DIM, 1)
        self.SAMPLE_SIZE = config.sample_size
        self.DIM = DIM

    def forward(self, input):
        input = input.view(-1, 1, self.SAMPLE_SIZE , self.SAMPLE_SIZE )
        out = self.main(input)
        out = out.view(-1, 4*4*8*self.DIM)
        out = self.output(out)
        return out.view(-1)


class Discriminator2(nn.Module):
    def __init__(self, config):
        super(Discriminator2, self).__init__()
        DIM = config.model_dim
        main = nn.Sequential(
            nn.Conv2d(2, DIM, 5, stride=2, padding=2), # [batch, DIM, 32, 32]
            nn.ReLU(True),
            nn.Conv2d(DIM, 2*DIM, 5, stride=2, padding=2),# [batch, 2*DIM, 16, 16]
            nn.ReLU(True),
            nn.Conv2d(2*DIM, 4*DIM, 5, stride=2, padding=2),# [batch, 4*DIM, 8, 8]
            nn.ReLU(True),
            nn.Conv2d(4 * DIM, 8*DIM, 5, stride=2, padding=2),  # [batch, 8*DIM, 4, 4]
            nn.ReLU(True),
        )
        self.main = main
        self.output = nn.Linear(4*4*8*DIM, 1)
        self.SAMPLE_SIZE = config.sample_size
        self.DIM = DIM

    def forward(self, input):
        out = self.main(input)
        out = out.view(-1, 4*4*8*self.DIM)
        out = self.output(out)
        return out.view(-1)