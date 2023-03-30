import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
import glob
import models.GAN_models as gan_models
import torch.optim as optim
import torch.autograd as autograd
import itertools
import h5py
import numpy as np


def thresh(data):
    data = data / 2 + 0.5
    data = data.reshape(data.size(0), -1)
    pos_pixel = torch.sum(data > 0.5, dim=1)
    entropy = torch.sum(data * torch.log(data + 1e-5), dim=1)
    select_idx = (pos_pixel > 64) &  (entropy > -70)
    return select_idx

class WhistleDataset4(): # Use wgan whose output range is [-1, 1] to generate A, Use labeled positive patches HDF5 as B
                        # and labeled negative patches HDF5 as B neg.
    def initialize(self, opt):
        self.opt = opt
        # self.root = opt.dataroot
        # if opt.pos_data_file is None:
        #     self.dir_AB = glob.glob(opt.dataroot + '/' + opt.phase + '*')
        # else:
        #     self.dir_AB = [opt.pos_data_file]

        self.dir_AB = [opt.pos_data_file]
        self.h5_dataset = HDF5_Dataset_transpose(self.dir_AB, opt.batchSize)

        # self.root_neg = opt.neg_dataroot
        # if opt.neg_data_file is None:
        #     self.dir_AB_neg = glob.glob(opt.neg_dataroot + '/' + opt.phase + '*')
        # else:
        #     self.dir_AB_neg = [opt.neg_data_file]
        self.dir_AB_neg = [opt.neg_data_file]
        self.h5_dataset_neg = HDF5_Dataset_transpose(self.dir_AB_neg, opt.batchSize)

        if opt.pos_G == "wgan":
            self.pos_Gnet = gan_models.wgan_Generator().cuda()
        if opt.pos_G == "wgan2":
            self.pos_Gnet = gan_models.wgan_Generator2().cuda()
        elif opt.pos_G == "dcgan":
            self.pos_Gnet = gan_models.dcgan_generator().cuda()
        else:
            raise ValueError("pos_G [%s] not recognized." % opt.pos_G)

        pos_model = opt.pos_Gnet_model
        model_dict = torch.load(pos_model)
        self.pos_Gnet.load_state_dict(model_dict['GNET'])
        self.pos_Gnet.eval()

        if opt.neg_G == "wgan":
            self.neg_Gnet = gan_models.wgan_Generator().cuda()
        elif opt.neg_G == "dcgan":
            self.neg_Gnet = gan_models.dcgan_generator().cuda()
        elif opt.neg_G == "wgan2":
            self.neg_Gnet = gan_models.wgan_Generator2().cuda()
        else:
            raise ValueError("neg_G [%s] not recognized." % opt.neg_G)

        neg_model = opt.neg_Gnet_model
        model_dict = torch.load(neg_model)
        self.neg_Gnet.load_state_dict(model_dict['GNET'])
        self.neg_Gnet.eval()

    def generate_image(self, netG, batchSize, n_latent, sample_size):
        with torch.no_grad():
            noise = torch.randn(batchSize, n_latent).cuda()
            # noisev = torch.autograd.Variable(noise, volatile=True)
            samples = netG(noise)
            samples = samples.detach()
        # samples = samples.view(batchSize, sample_size, sample_size)

        return samples

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.h5_dataset)

    def next(self):
        h5_batchdata = next(self.h5_dataset)
        B = h5_batchdata[0]
        B = torch.from_numpy(B).cuda()
        B = (B - 0.5) * 2
        B_label = h5_batchdata[1]
        B_label = torch.from_numpy(B_label).cuda()
        B_label = (B_label - 0.5) * 2

        h5_batchdata2 = next(self.h5_dataset_neg)
        B_neg = h5_batchdata2[0]
        B_neg = torch.from_numpy(B_neg).cuda()
        B_neg = (B_neg - 0.5) * 2

        batch_size = min(h5_batchdata[0].shape[0], self.opt.batchSize)
        count = 0
        pos_data = []
        while (count < batch_size):
            A_pos = self.generate_image(self.pos_Gnet, batch_size, self.opt.pos_n_latent, self.opt.fineSize)
            good_data = A_pos[thresh(A_pos),]
            count += good_data.size(0)
            pos_data.append(good_data)
        A_pos = torch.cat(pos_data, dim=0)
        A_pos = A_pos[:batch_size, ]
        A_neg = self.generate_image(self.neg_Gnet, batch_size, self.opt.neg_n_latent, self.opt.fineSize)
        A = torch.cat([A_pos, A_neg], dim=1)

        return {'A': A, 'B': B, 'B_label': B_label, 'B_neg': B_neg}

    def __next__(self):
        return self.next()

    def name(self):
        return 'WhistleDataset4'



def calc_gradient_penalty(netD, real_data, fake_data, gp_lambda=10):
    #print real_data.size()
    use_cuda = True
    alpha = torch.rand(int(real_data.size(0)), 1, 1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda() if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates.requires_grad_()

    disc_interpolates = netD(interpolates)
    temp = torch.ones(disc_interpolates.size()).cuda()
    # print(interpolates.requires_grad)
    # print(disc_interpolates.requires_grad)
    # print(temp.requires_grad)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=temp,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda
    return gradient_penalty


class WhistleDataset5_e2e():  # Use wgan whose output range is [-1, 1] to generate A, Use labeled positive patches HDF5 as B
                # and labeled negative patches HDF5 as B neg.
    def initialize(self, opt):
        self.opt = opt

        self.dir_AB = [opt.pos_data_file]
        self.h5_dataset = HDF5_Dataset_transpose(self.dir_AB, opt.batchSize)

        self.dir_AB_neg = [opt.neg_data_file]
        self.h5_dataset_neg = HDF5_Dataset_transpose(self.dir_AB_neg, opt.batchSize)

        if opt.pos_G == "wgan":
            self.pos_Gnet = gan_models.wgan_Generator().cuda()
        if opt.pos_G == "wgan2":
            self.pos_Gnet = gan_models.wgan_Generator2().cuda()
        elif opt.pos_G == "dcgan":
            self.pos_Gnet = gan_models.dcgan_generator().cuda()
        else:
            raise ValueError("pos_G [%s] not recognized." % opt.pos_G)
        self.pos_Dnet = gan_models.wgan_Discriminator().cuda()

        pos_model = opt.pos_Gnet_model
        model_dict = torch.load(pos_model)
        self.pos_Gnet.load_state_dict(model_dict['GNET'])
        self.pos_Dnet.load_state_dict(model_dict['DNET'])

        if opt.neg_G == "wgan":
            self.neg_Gnet = gan_models.wgan_Generator().cuda()
        elif opt.neg_G == "dcgan":
            self.neg_Gnet = gan_models.dcgan_generator().cuda()
        elif opt.neg_G == "wgan2":
            self.neg_Gnet = gan_models.wgan_Generator2().cuda()
        else:
            raise ValueError("neg_G [%s] not recognized." % opt.neg_G)
        self.neg_Dnet = gan_models.wgan_Discriminator().cuda()

        neg_model = opt.neg_Gnet_model
        model_dict = torch.load(neg_model)
        self.neg_Gnet.load_state_dict(model_dict['GNET'])
        self.neg_Dnet.load_state_dict(model_dict['DNET'])

        self.optimizerG = optim.Adam(itertools.chain(self.pos_Gnet.parameters(), self.neg_Gnet.parameters()), lr=1e-4, betas=(0.5, 0.9))
        self.optimizerD = optim.Adam(itertools.chain(self.pos_Dnet.parameters(), self.neg_Dnet.parameters()), lr=1e-4, betas=(0.5, 0.9))


    def generate_image(self, netG, batchSize, n_latent, no_grad=False):
        if no_grad:
            with torch.no_grad():
                noise = torch.randn(batchSize, n_latent).cuda()
                samples = netG(noise)
                samples = samples.detach()
        else:
            noise = torch.randn(batchSize, n_latent).cuda()
            samples = netG(noise)
            samples = samples.detach()
        return samples


    def __iter__(self):
        return self


    def __len__(self):
        return len(self.h5_dataset)


    def next(self):
        h5_batchdata = next(self.h5_dataset)
        B = h5_batchdata[0]
        B = torch.from_numpy(B).cuda()
        B = (B - 0.5) * 2
        B_label = h5_batchdata[1]
        B_label = torch.from_numpy(B_label).cuda()
        B_label = (B_label - 0.5) * 2

        h5_batchdata2 = next(self.h5_dataset_neg)
        B_neg = h5_batchdata2[0]
        B_neg = torch.from_numpy(B_neg).cuda()
        B_neg = (B_neg - 0.5) * 2

        batch_size = min(h5_batchdata[0].shape[0], self.opt.batchSize)
        A_pos = self.generate_image(self.pos_Gnet, batch_size, self.opt.pos_n_latent, False)
        A_neg = self.generate_image(self.neg_Gnet, batch_size, self.opt.neg_n_latent, False)
        A = torch.cat([A_pos, A_neg], dim=1)

        self.fake_pos = A_pos
        self.fake_neg = A_neg
        self.real_pos = B_label
        self.real_neg = B_neg
        return {'A': A, 'B': B, 'B_label': B_label, 'B_neg': B_neg}

    def D_loss(self, netD, real_data, fake_data):
        sample_size = 64
        batch_size = int(real_data.size(0))

        # train with real
        D_real = netD(real_data)
        D_real = D_real.mean()

        # train with fake
        D_fake = netD(fake_data)
        D_fake = D_fake.mean()

        # train with gradient penalty
        gradient_penalty = calc_gradient_penalty(netD, real_data, fake_data)

        D_cost = D_fake - D_real + gradient_penalty
        return D_cost


    def train_D(self):
        self.neg_Dnet.zero_grad()
        self.pos_Dnet.zero_grad()
        # D_loss_neg = self.D_loss(self.neg_Gnet, self.neg_Dnet, self.real_neg, use_cuda=True)
        # D_loss_pos = self.D_loss(self.pos_Gnet, self.pos_Dnet, self.real_pos, use_cuda=True)
        D_loss_neg = self.D_loss(self.neg_Dnet, self.real_neg, self.fake_neg)
        D_loss_pos = self.D_loss(self.pos_Dnet, self.real_pos, self.fake_pos)
        D_loss_neg.backward()
        D_loss_pos.backward()
        self.optimizerD.step()

    def G_loss(self, netD, fake_data):
        G = netD(fake_data)
        G = G.mean()
        G_cost = -G
        return G_cost

    def train_G(self):
        G_loss_pos = self.G_loss(self.pos_Dnet, self.fake_pos)
        G_loss_neg = self.G_loss(self.neg_Dnet, self.fake_pos)
        G_loss_pos.backward()
        G_loss_neg.backward()
        self.optimizerG.step()
        self.neg_Gnet.zero_grad()
        self.pos_Gnet.zero_grad()

    def save_networks(self, save_dir, which_epoch, name):
        save_filename = '%s_%s_WGAN.pth' % (which_epoch, name)
        save_path = os.path.join(save_dir, save_filename)
        if name == 'pos':
            netG = self.pos_Gnet
            netD = self.pos_Dnet
        elif name == 'neg':
            netG = self.neg_Gnet
            netD = self.neg_Dnet
        else:
            raise NotImplementedError
        states = {'GNET': netG.state_dict(),
                  'DNET': netD.state_dict()}
        torch.save(states, save_path)

    def __next__(self):
        return self.next()


    def name(self):
        return 'WhistleDataset5'



def read_h5_length(file):
    # print os.path.exists(file)
    h5file = h5py.File(file)
    length = len(h5file['data'])
    h5file.close()
    return length



def read_h5_pos(file, pos, nsamples):
    h5file = h5py.File(file)
    data = h5file['data'][pos:pos+nsamples]
    label = h5file['label'][pos:pos+nsamples]
    h5file.close()
    return data, label


class HDF5_Dataset_transpose():
    def __init__(self, hdf5_list, batchsize):
        self.hdf5_list = hdf5_list
        self.nfiles = len(hdf5_list)
        self.batch_size = batchsize
        self.epoches = 0
        self.curr_file = 0
        self.curr_file_pointer = 0

        length_list = list(map(read_h5_length, hdf5_list))
        self.len_list = length_list
        self.total_count = np.sum(length_list)

    def __len__(self):
        print(self.hdf5_list)
        print(self.total_count)
        return self.total_count

    def __iter__(self):
        return self

    def next(self):
        h5_file = self.hdf5_list[self.curr_file]
        data, label = read_h5_pos(h5_file, self.curr_file_pointer, self.batch_size)
        data = np.transpose(data, (0, 1, 3, 2))
        label = np.transpose(label, (0, 1, 3, 2))
        self.curr_file_pointer += self.batch_size
        if self.curr_file_pointer >= self.len_list[self.curr_file]:
            self.curr_file_pointer = 0
            self.curr_file += 1
            if self.curr_file + 1 > self.nfiles:
                self.curr_file = 0
                self.epoches += 1
        return data, label

    def __next__(self):
        return self.next()

