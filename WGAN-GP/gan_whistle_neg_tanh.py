import os, sys
sys.path.append(os.getcwd())

import time

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import utils.m_dataset as m_dataset
import utils.m_func as m_func

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets

import tflib as lib
import tflib.save_images
import tflib.plot


torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
if use_cuda:
    gpu = 0

exp_name = 'whistle_neg_tanh2'
test_dir = './test_result/' + exp_name
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

model_dir = './models/' + exp_name
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

n_save_model = 10000
# SAMPLE_SIZE = 28
SAMPLE_SIZE = 64
DIM = 64 # Model dimensionality
BATCH_SIZE = 64 # Batch size
CRITIC_ITERS = 5 # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10 #10 # Gradient penalty lambda hyperparameter
ITERS = 100000 # How many generator iterations to train for
OUTPUT_DIM = SAMPLE_SIZE * SAMPLE_SIZE # Number of pixels in MNIST (28*28) 64*64

lib.print_model_settings(locals().copy())


class Generator(nn.Module):# ==================Definition Start======================
    def __init__(self):
        super(Generator, self).__init__()

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
        conv_out = nn.Conv2d(DIM/2, 4, 3, padding=1)

        self.block1 = block1
        self.block2 = block2
        self.block3 = block3
        self.conv_out = conv_out
        self.preprocess = preprocess
        self.Tanh = nn.Tanh()

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 4*DIM, 4, 4) # [batch, 4*DIM, 4, 4]
        output = self.block1(output) # [batch, 8*DIM, 4, 4]
        output = nn.functional.pixel_shuffle(output, 2) # [batch, 2*DIM, 8, 8]
        output = self.block2(output) # [batch, 4*DIM, 8, 8]
        output = nn.functional.pixel_shuffle(output, 2)  # [batch, DIM, 16, 16]
        output = self.block3(output)  # [batch, 2*DIM, 16, 16]
        output = nn.functional.pixel_shuffle(output, 2)  # [batch, DIM/2, 32, 32]
        output = self.conv_out(output) # [batch, 4, 32, 32]
        output = nn.functional.pixel_shuffle(output, 2)  # [batch, 1, 64, 64]
        output = self.Tanh(output)
        return output.view(-1, OUTPUT_DIM)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

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

    def forward(self, input):
        input = input.view(-1, 1, SAMPLE_SIZE, SAMPLE_SIZE)
        out = self.main(input)
        out = out.view(-1, 4*4*8*DIM)
        out = self.output(out)
        return out.view(-1)

def generate_image(frame, netG):
    noise = torch.randn(BATCH_SIZE, 128)
    if use_cuda:
        noise = noise.cuda(gpu)
    noisev = autograd.Variable(noise, volatile=True)
    samples = netG(noisev)

    samples = (samples + 1) / 2
    samples = samples.view(BATCH_SIZE, SAMPLE_SIZE, SAMPLE_SIZE)
    # print samples.size()

    samples = samples.cpu().data.numpy()

    lib.save_images.save_images(
        samples,
        test_dir + '/samples_{}.png'.format(frame)
    )


def calc_gradient_penalty(netD, real_data, fake_data):
    #print real_data.size()
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda(gpu) if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda(gpu)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(gpu) if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

# ==================Definition End======================
def removeLineFeed(line):
    return line[:-1]

def read_lmdb_list(file):
    with open(file) as f:
        data = f.readlines()
    data = map(removeLineFeed, data)
    return data

train_file = 'train_neg.txt'
train_filelist = read_lmdb_list(train_file)

test_file = 'test.txt'
test_filelist = read_lmdb_list(test_file)

train_dataset = m_dataset.HDF5_Dataset_transpose(hdf5_list=train_filelist, batchsize=BATCH_SIZE)
test_dataset = m_dataset.HDF5_Dataset_transpose(hdf5_list=test_filelist, batchsize=BATCH_SIZE)

netG = Generator()
netD = Discriminator()
netG.apply(m_func.weights_init_He_normal)
netD.apply(m_func.weights_init_He_normal)
print netG
print netD

if use_cuda:
    netD = netD.cuda(gpu)
    netG = netG.cuda(gpu)

optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))

one = torch.FloatTensor([1])
mone = one * -1
if use_cuda:
    one = one.cuda(gpu)
    mone = mone.cuda(gpu)

for iteration in xrange(ITERS):
    start_time = time.time()
    ############################
    # (1) Update D network
    ###########################
    for p in netD.parameters():  # reset requires_grad
        p.requires_grad = True  # they are set to False below in netG update

    for iter_d in xrange(CRITIC_ITERS):
        _data, _label = next(train_dataset)
        _data = _data[:,:,:SAMPLE_SIZE, :SAMPLE_SIZE]
        _data = (_data - 0.5) * 2
        real_data = torch.Tensor(_data)
        real_data = real_data.contiguous().view(-1, OUTPUT_DIM)
        if use_cuda:
            real_data = real_data.cuda(gpu)
        real_data_v = autograd.Variable(real_data)

        netD.zero_grad()

        # train with real
        D_real = netD(real_data_v)
        D_real = D_real.mean()
        # print D_real
        D_real.backward(mone)

        # train with fake
        noise = torch.randn(BATCH_SIZE, 128)
        if use_cuda:
            noise = noise.cuda(gpu)
        noisev = autograd.Variable(noise, volatile=True)  # totally freeze netG
        fake = autograd.Variable(netG(noisev).data)
        inputv = fake
        D_fake = netD(inputv)
        D_fake = D_fake.mean()
        D_fake.backward(one)

        # train with gradient penalty
        gradient_penalty = calc_gradient_penalty(netD, real_data_v.data, fake.data)
        gradient_penalty.backward()

        D_cost = D_fake - D_real + gradient_penalty
        Wasserstein_D = D_real - D_fake
        optimizerD.step()

    ############################
    # (2) Update G network
    ###########################
    for p in netD.parameters():
        p.requires_grad = False  # to avoid computation
    netG.zero_grad()

    noise = torch.randn(BATCH_SIZE, 128)
    if use_cuda:
        noise = noise.cuda(gpu)
    noisev = autograd.Variable(noise)
    fake = netG(noisev)
    G = netD(fake)
    G = G.mean()
    G.backward(mone)
    G_cost = -G
    optimizerG.step()

    # Write logs and save samples
    lib.plot.plot(test_dir + '/time', time.time() - start_time)
    lib.plot.plot(test_dir + '/train disc cost', D_cost.cpu().data.numpy())
    lib.plot.plot(test_dir + '/train gen cost', G_cost.cpu().data.numpy())
    lib.plot.plot(test_dir + '/wasserstein distance', Wasserstein_D.cpu().data.numpy())

    # Calculate dev loss and generate samples every 100 iters
    if iteration % 1000 == 0:
        generate_image(iteration, netG)

    # Write logs every 100 iters
    if (iteration < 5) or (iteration % 1000 == 0):
        lib.plot.flush()


    if iteration % n_save_model == 0:
        print("Saving model ...")
        states = {'GNET': netG.state_dict(),
                  'DNET': netD.state_dict()}
        torch.save(states, model_dir + '/WGAN_iter' + str(iteration))

    lib.plot.tick()
