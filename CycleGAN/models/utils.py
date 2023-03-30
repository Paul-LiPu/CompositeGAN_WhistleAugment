import os
import sys
curr_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(curr_path)
from . import GAN_models as gan
import torch
from options.data_options import Data_Options


def create_wgan(type):
    if type == "wgan":
        Gnet = gan.wgan_Generator().cuda()
    elif type == "dcgan":
        Gnet = gan.dcgan_generator().cuda()
    elif type == "wgan2":
        Gnet = gan.wgan_Generator2().cuda()
    else:
        raise ValueError("Type [%s] not recognized." % type)
    return Gnet

def generate_image(netG, batchSize, no_grad=True):
    n_latent = netG.input_dim
    if no_grad:
        with torch.no_grad():
            noise = torch.randn(batchSize, n_latent)
            noise = noise.cuda()
            samples = netG(noise)
            samples = samples.detach()
    else:
        noise = torch.randn(batchSize, n_latent)
        noise = noise.cuda()
        samples = netG(noise)
        samples = samples.detach()
    # samples = samples.view(batchSize, sample_size, sample_size)
    return samples

def shift_value(image, reverse=False):
    if reverse:
        image = (image - 0.5) * 2
    else:
        image = image / 2 + 0.5
    return image

def create_cyclegan(type, checkpoints_dir, name):
    if type == 'whistle_cycle_deepContour_negID_residual':
        from gan_models.whistle_cycleGAN_deepcontour_negId_model_residual import WhistleCycleGANDeepContourNegIDModel_Res
        model = WhistleCycleGANDeepContourNegIDModel_Res()
    else:
        raise NotImplementedError('model [%s] not implemented.' % type)
    opt = Data_Options().parse()
    opt.netG_A_residual = True
    opt.netG_A_rlow = 0.1
    opt.netG_A_rhigh = 1
    opt.isTrain = False
    opt.checkpoints_dir = checkpoints_dir
    opt.name = name
    model.initialize(opt)
    return model


def generate_fakeB(model, pos, neg):
    inputA = torch.cat([pos, neg], dim=1)
    model.input_A = inputA
    with torch.no_grad():
        model.test()
    model.fake_B.detach()
    return model.fake_B