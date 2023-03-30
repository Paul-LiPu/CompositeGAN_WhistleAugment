# -*- coding: utf-8 -*-
# @Time    : 9/17/20 7:23 PM
# @Author  : Pu Li
# @Email   : pli5270@sdsu.edu
# @File    : train.py
# @Description:

# import library
import os, sys
sys.path.append(os.getcwd())
import time
import torch
import torch.autograd as autograd
import torch.optim as optim
import utils.m_dataset as m_dataset
import utils.m_func as m_func
from utils.model import Generator, Discriminator
from utils import util
import matplotlib
import tflib as lib
import tflib.save_images
import tflib.plot
from utils import arg_parser
from logger import getLogger

# preset of the library.
matplotlib.use('Agg')
torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
if use_cuda:
    gpu = 0

# Read Arguments.
parser = arg_parser.ArgParser()
config = parser.parse()
#
exp_name = config.exp_name
test_dir = os.path.join(config.log_dir, exp_name)
img_dir = os.path.join(test_dir, 'imgs')
config.test_dir = test_dir
config.img_dir = img_dir
if not os.path.exists(test_dir):
    os.makedirs(test_dir)
if not os.path.exists(img_dir):
    os.makedirs(img_dir)

timestamp = time.strftime("%Y%m%d_%H-%M-%S", time.localtime())
logFile = os.path.join(test_dir, exp_name + '_' + timestamp + '.log')
txt_logger = getLogger(logFile)

model_dir = os.path.join(config.model_dir, exp_name)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

txt_logger.info(config)

# n_save_model = config.niter_save_model
# SAMPLE_SIZE = config.sample_size
# DIM = 64 # Model dimensionality
# BATCH_SIZE = 64 # Batch size
# CRITIC_ITERS = 5 # For WGAN and WGAN-GP, number of critic iters per gen iter
# LAMBDA = 10 #10 # Gradient penalty lambda hyperparameter
# ITERS = 100000 # How many generator iterations to train for
# OUTPUT_DIM = SAMPLE_SIZE * SAMPLE_SIZE # Number of pixels in MNIST (28*28) 64*64

lib.print_model_settings(locals().copy())

# ==================Definition Start=====================
def generate_image(frame, netG, config):
    noise = torch.randn(config.batch_size, 128)
    if use_cuda:
        noise = noise.cuda(gpu)
    with torch.no_grad():
        samples = netG(noise)

    samples = (samples + 1) / 2
    samples = samples.view(config.batch_size, config.sample_size, config.sample_size)
    # print samples.size()

    samples = samples.cpu().data.numpy()

    lib.save_images.save_images(
        samples,
        config.img_dir + '/samples_{}.png'.format(frame)
    )


def calc_gradient_penalty(netD, real_data, fake_data, config):
    #print real_data.size()
    alpha = torch.rand(int(real_data.size(0)), 1)
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

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * config.gp_lambda
    return gradient_penalty

train_filelist = util.read_array_file(config.train_file, str)
train_filelist = train_filelist.squeeze(axis=1).tolist()
# train_filelist = train_filelist.tolist()
# test_filelist = util.read_array_file(config.test_file, str).squeeze().tolist()

train_dataset = m_dataset.HDF5_Dataset_transpose(hdf5_list=train_filelist, batchsize=config.batch_size)
# test_dataset = m_dataset.HDF5_Dataset_transpose(hdf5_list=test_filelist, batchsize=config.batch_size)

netG = Generator(config)
netD = Discriminator(config)
netG.apply(m_func.weights_init_He_normal)
netD.apply(m_func.weights_init_He_normal)
txt_logger.info(netG)
txt_logger.info(netD)

temp1 = len(train_dataset)
epoch_size = len(train_dataset) // (config.batch_size * config.critic_iter)

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

epoch = 0
time_cost_cum = 0
for iteration in range(config.train_iters):
    start_time = time.time()
    ############################
    # (1) Update D network
    ###########################
    for p in netD.parameters():  # reset requires_grad
        p.requires_grad = True  # they are set to False below in netG update

    D_cost_sum = 0
    Wasserstein_D_sum = 0
    for iter_d in range(config.critic_iter):
        _data, _label = next(train_dataset)
        if config.mode == 'pos':
            _label = _label[:,:,:config.sample_size, :config.sample_size]
            _label = (_label - 0.5) * 2
            real_data = torch.Tensor(_label)
        else:
            _data = _data[:, :, :config.sample_size, :config.sample_size]
            _data = (_data - 0.5) * 2
            real_data = torch.Tensor(_data)
        real_data = real_data.contiguous().view(-1, config.sample_size ** 2)

        if use_cuda:
            real_data = real_data.cuda(gpu)
        batch_size = int(real_data.size(0))
        real_data_v = autograd.Variable(real_data)

        netD.zero_grad()

        # train with real
        D_real = netD(real_data_v)
        D_real = D_real.mean()
        # print D_real
        # D_real.backward(mone)

        # train with fake
        noise = torch.randn(batch_size, 128)
        if use_cuda:
            noise = noise.cuda(gpu)
        with torch.no_grad():
            fake = netG(noise)
        inputv = fake
        D_fake = netD(inputv)
        D_fake = D_fake.mean()
        # D_fake.backward(one)

        # train with gradient penalty
        gradient_penalty = calc_gradient_penalty(netD, real_data_v.data, fake.data, config)
        # gradient_penalty.backward()

        D_cost = D_fake - D_real + gradient_penalty
        D_cost.backward()
        Wasserstein_D = D_real - D_fake
        optimizerD.step()
        D_cost_sum += D_cost.cpu().data.numpy()
        Wasserstein_D_sum += Wasserstein_D.cpu().data.numpy()

    ############################
    # (2) Update G network
    ###########################
    for p in netD.parameters():
        p.requires_grad = False  # to avoid computation
    netG.zero_grad()

    noise = torch.randn(config.batch_size, 128)
    if use_cuda:
        noise = noise.cuda(gpu)
    noisev = autograd.Variable(noise)
    fake = netG(noisev)
    G = netD(fake)
    G = G.mean()
    # G.backward(mone)
    G_cost = -G
    G_cost.backward()
    optimizerG.step()

    # Write logs and save samples
    time_cost = time.time() - start_time
    time_cost_cum += time_cost
    avg_D_cost = D_cost_sum / config.critic_iter
    G_cost_np = G_cost.cpu().data.numpy()
    avg_Wasserstein_D = Wasserstein_D_sum / config.critic_iter
    lib.plot.plot(config.img_dir + '/time', time_cost)
    lib.plot.plot(config.img_dir + '/train disc cost', avg_D_cost)
    lib.plot.plot(config.img_dir + '/train gen cost', G_cost_np)
    lib.plot.plot(config.img_dir + '/wasserstein distance', avg_Wasserstein_D)

    if (iteration+1) % epoch_size == 0:
        epoch += 1

    if (iteration+1) % 10 == 0:
        info = 'D_cost: %s, G_cost: %s, Wasserstein_D: %s' % (str(avg_D_cost), str(G_cost_np),  str(avg_Wasserstein_D))
        txt_logger.info('[epoch %s iter %s]: lr: %s %s time: %s' % (
            str(epoch), str(iteration), str(1e-4), info, str(time_cost_cum)))
        time_cost_cum = 0
        lib.plot.flush()

    # Calculate dev loss and generate samples every 100 iters
    if (iteration+1) % 1000 == 0:
        generate_image(iteration, netG, config)

    # Write logs every 100 iters
    # if (iteration % 10 == 0):

    if (iteration+1) % config.niter_save_model == 0:
        txt_logger.info("Saving model ...")
        states = {'GNET': netG.state_dict(),
                  'DNET': netD.state_dict()}
        torch.save(states, model_dir + '/WGAN_iter' + str(iteration))

    lib.plot.tick()
