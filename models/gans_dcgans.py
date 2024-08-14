import torch
import torch.nn as nn


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self, img_size, latent_dim, channels):
        super(Generator, self).__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
            # nn.Sigmoid(),
        )

        # NOTE 우리 방법만의 adapter
        # self.adapter = nn.Conv2d(64, vig_channels, 1)

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        # out.shape == torch.Size([5, 128, 8, 8])
        out = self.conv_blocks[:3](out)
        # out.shape == torch.Size([5, 128, 16, 16])
        out = self.conv_blocks[3:6](out)
        # out.shape == torch.Size([5, 128, 32, 32])
        adapt_out = out = self.conv_blocks[6:9](out)
        # out.shape == torch.Size([5, 64, 32, 32])
        # NOTE 마지막에 nn.Tanh() 를 통과시키지 않아서 문제가 된 적이 있음
        out = self.conv_blocks[9:](out)
        # out.shape == torch.Size([5, 3, 32, 32])
        # return out, self.adapter(adapt_out)
        return out, adapt_out


class Discriminator(nn.Module):
    def __init__(self, img_size, channels):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(
                0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Sequential(
            nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


class ManageGANs:
    def __init__(self, imgsize, ngf, nc, device):
        """
        Parameters:
            size of the noise
        ngf: int
            size of hidden states in generator
        nc: int
            number of generated channels
        ndf: int
            size of hidden states in discriminator

        """
        self.netG = Generator(imgsize, ngf, nc).to(device)
        self.netD = Discriminator(imgsize, nc).to(device)
        # NOTE 반드시 이렇게 weight 초기화를 해야 함
        self.netG.apply(weights_init_normal)
        self.netD.apply(weights_init_normal)
        self.ngf = ngf
        self.nc = nc
        self.device = device

    def get_GD(self):
        return self.netG, self.netD

    def get_noise(self, size):
        return torch.FloatTensor(size, self.ngf).normal_(0, 1).to(self.device)
