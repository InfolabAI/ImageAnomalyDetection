# reference code is https://github.com/pytorch/examples/blob/master/dcgan/main.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from models import *


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    # elif classname.find('Linear') != -1:
    #    m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class _netD(nn.Module):
    def __init__(self, nc, ndf):
        super(_netD, self).__init__()
        self.main = nn.Sequential(
            # input size. (nc) x 32 x 32
            # nn.Conv2d(nc, ndf * 2, 4, 2, 1, bias=False),
            nn.Linear(nc, ndf * 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            # nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.Linear(ndf * 2, ndf * 4, bias=False),
            nn.BatchNorm1d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            # nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.Linear(ndf * 4, ndf * 8, bias=False),
            nn.BatchNorm1d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            # nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Linear(ndf * 8, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)

        return output.view(-1, 1)


class _netG(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(_netG, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            # nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.Linear(nz, ngf * 8, bias=False),
            nn.BatchNorm1d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            # nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.Linear(ngf * 8, ngf * 4, bias=False),
            nn.BatchNorm1d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            # nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.Linear(ngf * 4, ngf * 2, bias=False),
            nn.BatchNorm1d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            # nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
            nn.Linear(ngf * 2, nc, bias=False),
            nn.Sigmoid()
            # state size. (nc) x 32 x 32
        )

    def forward(self, input):
        output = self.main(input)
        return output


def Generator(nz, ngf, nc):
    model = _netG(nz, ngf, nc)
    model.apply(weights_init)
    return model


def Discriminator(nc, ndf):
    model = _netD(nc, ndf)
    model.apply(weights_init)
    return model


class ManageGANs_Linear:
    def __init__(self, nz, ngf, nc, ndf, device):
        """
        Parameters:
        -----------
        nz: int
            size of the noise
        ngf: int
            size of hidden states in generator
        nc: int
            number of generated channels
        ndf: int
            size of hidden states in discriminator

        """
        self.netG = Generator(nz, ngf, nc).to(device)
        self.netD = Discriminator(nc, ndf).to(device)
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        self.ndf = ndf
        self.device = device

    def get_GD(self):
        return self.netG, self.netD

    def get_noise(self, size):
        return torch.FloatTensor(size, self.nz).normal_(0, 1).to(self.device)
