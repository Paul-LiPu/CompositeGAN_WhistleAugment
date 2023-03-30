import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class wgan_Generator(nn.Module):
    def __init__(self, DIM=64):
        self.input_dim = 128
        self.dim = DIM
        super(wgan_Generator, self).__init__()

        preprocess = nn.Sequential(
            nn.Linear(self.input_dim, 4*4*4*DIM),
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
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 4*self.dim, 4, 4) # [batch, 4*DIM, 4, 4]
        output = self.block1(output) # [batch, 8*DIM, 4, 4]
        output = nn.functional.pixel_shuffle(output, 2) # [batch, 2*DIM, 8, 8]
        output = self.block2(output) # [batch, 4*DIM, 8, 8]
        output = nn.functional.pixel_shuffle(output, 2)  # [batch, DIM, 16, 16]
        output = self.block3(output)  # [batch, 2*DIM, 16, 16]
        output = nn.functional.pixel_shuffle(output, 2)  # [batch, DIM/2, 32, 32]
        output = self.conv_out(output) # [batch, 4, 32, 32]
        output = nn.functional.pixel_shuffle(output, 2)  # [batch, 1, 64, 64]
        output = self.sigmoid(output)

        return output




class wgan_Generator2(nn.Module):
    def __init__(self, DIM=64):
        self.input_dim = 128
        self.dim = DIM
        super(wgan_Generator2, self).__init__()

        preprocess = nn.Sequential(
            nn.Linear(self.input_dim, 4*4*4*DIM),
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

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 4*self.dim, 4, 4) # [batch, 4*DIM, 4, 4]
        output = self.block1(output) # [batch, 8*DIM, 4, 4]
        output = nn.functional.pixel_shuffle(output, 2) # [batch, 2*DIM, 8, 8]
        output = self.block2(output) # [batch, 4*DIM, 8, 8]
        output = nn.functional.pixel_shuffle(output, 2)  # [batch, DIM, 16, 16]
        output = self.block3(output)  # [batch, 2*DIM, 16, 16]
        output = nn.functional.pixel_shuffle(output, 2)  # [batch, DIM/2, 32, 32]
        output = self.conv_out(output) # [batch, 4, 32, 32]
        output = nn.functional.pixel_shuffle(output, 2)  # [batch, 1, 64, 64]
        output = self.Tanh(output)

        return output


class wgan_Generator3(nn.Module):
    def __init__(self, DIM=64):
        super(wgan_Generator3, self).__init__()
        # self.dim = DIM
        self.input_dim = 128
        preprocess = nn.Sequential(
            nn.Linear(self.input_dim, 4*4*8*DIM),
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


class wgan_Discriminator(nn.Module):
    def __init__(self, DIM=64):
        super(wgan_Discriminator, self).__init__()
        self.dim = DIM

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
        out = self.main(input)
        out = out.view(-1, 4*4*8*self.dim)
        out = self.output(out)

        return out

class dcgan_generator(nn.Module):
    # initializers
    def __init__(self, d=128):

        super(dcgan_generator, self).__init__()
        self.input_dim = 100
        self.deconv1 = nn.ConvTranspose2d(self.input_dim, d*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 1, 4, 2, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        # x = F.relu(self.deconv1(input))
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = F.tanh(self.deconv5(x))

        return x

class dcgan_discriminator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(dcgan_discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8, 1, 4, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.sigmoid(self.conv5(x))

        return x