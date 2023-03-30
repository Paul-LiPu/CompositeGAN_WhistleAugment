import torch
from torch.autograd import Variable
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from torch.autograd import Variable
import torch.nn as nn
import numpy as np


class WhistleCycleGANDeepContourNegIDModel_Res(BaseModel):
    def name(self):
        return 'WhistleCycleGANDeepContourNegIDModelResidual'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['D_A', 'D_B_pos', 'D_B_neg', 'G_A', 'G_B_pos', 'G_B_neg', 'cycle_A_pos',
                           'cycle_A_neg', 'cycle_B', 'fake_B_confidence', 'fake_A_neg_Idt', 'fake_B_neg_Idt']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        if opt.netG_A_residual:
            visual_names_A = ['real_A', 'fake_B_res', 'fake_B', 'rec_A']
        else:
            visual_names_A = ['real_A', 'fake_B', 'rec_A']
        # visual_names_B = ['real_B', 'fake_A', 'rec_B']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        # if self.isTrain and self.opt.lambda_identity > 0.0:
        #     visual_names_A.append('idt_A')
        #     visual_names_B.append('idt_B')

        # self.visual_names = visual_names_A + visual_names_B
        self.visual_names = visual_names_A + visual_names_B
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B_pos', 'D_B_neg']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        self.contour_net = Detection_ResNet_BN2(width=32).cuda(self.gpu_ids[0])
        if opt.contour_model != '':
            self.contour_net.load_state_dict(torch.load(opt.contour_model))

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            self.netD_B_pos = networks.define_D(opt.input_cycle_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            self.netD_B_neg = networks.define_D(opt.input_cycle_nc, opt.ndf,
                                                opt.which_model_netD,
                                                opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B_pos.parameters(),
                                                                self.netD_B_neg.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.which_epoch)
        self.print_networks(opt.verbose)

        self.zero = Variable(torch.zeros(self.opt.batchSize, 1, self.opt.loadSize, self.opt.loadSize), volatile=True).cuda()
        self.opt = opt
        self.netG_A_residual = opt.netG_A_residual
        self.netG_A_rlow = opt.netG_A_rlow
        self.netG_A_rhigh = opt.netG_A_rhigh

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        self.real_B_neg = input['B_neg']
        self.input_A = input_A
        self.input_B = input_B
        # self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_A = self.input_A
        self.real_B = self.input_B

    def forward_residual(self, net, input, orig):
        low = self.netG_A_rlow
        high = self.netG_A_rhigh
        # setting for 4species_whistle_cycle_deepContour_negID_residual
        # random_weight = np.random.random() * (high - low) + low

        # setting for 4species_whistle_cycle_deepContour_negID_residual2

        output = net(input)
        random_weight = torch.rand(output.size(0), 1, 1, 1) * (high - low) + low
        random_weight = random_weight.cuda()
        return output * random_weight + orig, output


    def test(self):
        self.real_A = self.input_A
        if self.netG_A_residual:
            self.fake_B, self.fake_B_res = self.forward_residual(self.netG_A, self.real_A, self.real_A[:, 1:, :, :])
        else:
            self.fake_B = self.netG_A(self.real_A)
        self.rec_A = self.netG_B(self.fake_B)

        # self.real_B = Variable(self.input_B, volatile=True)
        # self.fake_A = self.netG_B(self.real_B)
        # self.rec_B = self.netG_A(self.fake_A)

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        # fake_B = self.fake_B_pool.query(self.fake_B)
        fake_B = self.fake_B
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        # fake_A = self.fake_A_pool.query(self.fake_A)
        fake_A = self.fake_A
        self.loss_D_B_pos = self.backward_D_basic(self.netD_B_pos, self.real_A[:,:1,:,:], fake_A[:,:1,:,:])
        self.loss_D_B_neg = self.backward_D_basic(self.netD_B_neg, self.real_A[:,1:2,:,:], fake_A[:,1:2,:,:])

    def backward_G(self):
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B

        # GAN loss D_A(G_A(A))
        if self.netG_A_residual:
            self.fake_B, self.fake_B_res = self.forward_residual(self.netG_A, self.real_A, self.real_A[:, 1:, :, :])
        else:
            self.fake_B = self.netG_A(self.real_A)
        # self.fake_B = self.netG_A(self.real_A)
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)

        # GAN loss D_B(G_B(B))
        self.fake_A = self.netG_B(self.real_B)
        self.loss_G_B_pos = self.criterionGAN(self.netD_B_pos(self.fake_A[:,:1,:,:]), True)
        self.loss_G_B_neg = self.criterionGAN(self.netD_B_neg(self.fake_A[:,1:2,:,:]), True)

        # Forward cycle loss
        self.rec_A = self.netG_B(self.fake_B)
        self.loss_cycle_A_pos = self.criterionCycle(self.rec_A[:, :1, :, :], self.real_A[:, :1, :, :]) * lambda_A
        self.loss_cycle_A_neg = self.criterionCycle(self.rec_A[:, 1:2, :, :], self.real_A[:, 1:2, :, :]) * lambda_A

        # Backward cycle loss
        if self.netG_A_residual:
            self.rec_B, self.rec_B_res = self.forward_residual(self.netG_A, self.fake_A, self.fake_A[:, 1:, :, :])
            self.rec_B = self.rec_B_res + self.fake_A[:, 1:, :, :]
        else:
            self.rec_B = self.netG_A(self.real_A)
        # self.rec_B = self.netG_A(self.fake_A)
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        # Use contour net as a regulation for spatial position of added whistle
        self.fake_B_confidence = self.contour_net((self.fake_B + 1) / 2.0)
        self.loss_fake_B_confidence = self.criterionIdt(self.fake_B_confidence, (self.real_A[:, :1, :, :] + 1) / 2.0) * self.opt.LAMBDA_CONFIDENCE

        # Output identity negative patch as regulation for output overall intensity
        self.real_A_neg = self.real_A.clone()
        self.real_A_neg[:, 0, :, :] = -1
        if self.netG_A_residual:
            self.fake_B_neg, self.fake_B_neg_res = self.forward_residual(self.netG_A, self.real_A_neg, self.real_A_neg[:, 1:, :, :])
        else:
            self.fake_B_neg = self.netG_A(self.real_A)
        self.loss_fake_B_neg_Idt = self.criterionIdt(self.fake_B_neg, self.real_A_neg[:, 1:, :, :])
        self.loss_fake_B_neg_Idt *= self.opt.LAMBDA_B_neg

        self.fake_A_neg = self.netG_B(self.real_B_neg)

        mones = Variable(torch.ones(self.real_B_neg.size(0), 1, self.opt.loadSize, self.opt.loadSize),
                             volatile=True).cuda() * -1
        self.loss_fake_A_neg_Idt = self.criterionIdt(self.fake_A_neg[:, 1:, :, :], self.real_B_neg) + self.criterionIdt(self.fake_A_neg[:, :1, :, :], mones)
        self.loss_fake_A_neg_Idt *= self.opt.LAMBDA_A_neg

        # combined loss
        # self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G = self.loss_G_A + self.loss_G_B_pos + self.loss_G_B_neg + self.loss_cycle_A_pos + self.loss_cycle_A_neg + \
                      self.loss_cycle_B + self.loss_fake_B_confidence + self.loss_fake_A_neg_Idt + self.loss_fake_B_neg_Idt
        self.loss_G.backward()

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_A and D_B
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()

    def get_current_visuals(self):
        from collections import OrderedDict
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                if 'A' in name:
                    attrs = getattr(self, name)
                    visual_ret[name + '_front'] = attrs[:, :1, :, :]
                    visual_ret[name + '_back'] = attrs[:, 1:, :, :]
                else:
                    visual_ret[name] = getattr(self, name)
        return visual_ret


class Detection_ResNet_BN2(nn.Module):
    def __init__(self, width):
        super(Detection_ResNet_BN2, self).__init__()
        self.conv1 = nn.Conv2d(1, width, 5, padding=2)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.conv2_bn = nn.BatchNorm2d(num_features=width, eps=1e-05, momentum=0.1, affine=True)
        self.relu2 = nn.PReLU(num_parameters=width)
        self.conv3 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.conv3_bn = nn.BatchNorm2d(num_features=width, eps=1e-05, momentum=0.1, affine=True)
        self.conv4 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.conv4_bn = nn.BatchNorm2d(num_features=width, eps=1e-05, momentum=0.1, affine=True)
        self.relu4 = nn.PReLU(num_parameters=width)
        self.conv5 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.conv5_bn = nn.BatchNorm2d(num_features=width, eps=1e-05, momentum=0.1, affine=True)
        self.conv6 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.conv6_bn = nn.BatchNorm2d(num_features=width, eps=1e-05, momentum=0.1, affine=True)
        self.relu6 = nn.PReLU(num_parameters=width)
        self.conv7 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.conv7_bn = nn.BatchNorm2d(num_features=width, eps=1e-05, momentum=0.1, affine=True)
        self.conv8 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.conv8_bn = nn.BatchNorm2d(num_features=width, eps=1e-05, momentum=0.1, affine=True)
        self.relu8 = nn.PReLU(num_parameters=width)
        self.conv9 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.conv9_bn = nn.BatchNorm2d(num_features=width, eps=1e-05, momentum=0.1, affine=True)
        self.conv10 = nn.Conv2d(width, 1, 3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x_res = self.conv3_bn(self.conv3(self.relu2(self.conv2_bn(self.conv2(x)))))
        x = x + x_res
        x_res = self.conv5_bn(self.conv5(self.relu4(self.conv4_bn(self.conv4(x)))))
        x = x + x_res
        x_res = self.conv7_bn(self.conv7(self.relu6(self.conv6_bn(self.conv6(x)))))
        x = x + x_res
        x_res = self.conv9_bn(self.conv9(self.relu8(self.conv8_bn(self.conv8(x)))))
        x = x + x_res
        x = self.conv10(x)
        return x