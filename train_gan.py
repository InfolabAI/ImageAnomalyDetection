import argparse
import PIL
import tqdm
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from models.gans_dcgans import ManageGANs

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--data_name", type=str, default="")
parser.add_argument("--n_epochs", type=int, default=200000,
                    help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64,
                    help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002,
                    help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8,
                    help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100,
                    help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32,
                    help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3,
                    help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=1000,
                    help="interval between image sampling")
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False
data_config = {
    "lg_PCBNG_0.1_M": ["/home/robert.lim/datasets/PCB/TargetTVTReduced01_modified/train/0"],
    "lg_PCBNG_0.01_M": ["/home/robert.lim/datasets/PCB/TargetTVTReduced001_modified/train/0"],
    "mvtec_capsule": ["/home/robert.lim/datasets/mvtec/capsule/train/good"],
    "mvtec_capsule+pill": ["/home/robert.lim/datasets/mvtec/bottle/train/good", "/home/robert.lim/datasets/mvtec/pill/train/good"]
}
data_name = opt.data_name


# Loss function
adversarial_loss = torch.nn.BCELoss()
device = torch.device("cuda")
# Initialize generator and discriminator
generator, discriminator = ManageGANs(
    imgsize=32, ngf=opt.latent_dim, nc=3, device=device).get_GD()

generator.to(device)
discriminator.to(device)
adversarial_loss.to(device)

if os.path.exists(f"saved_model/gan/{data_name}/G.pth"):
    generator.load_state_dict(torch.load(
        f"saved_model/gan/{data_name}/G.pth"))
if os.path.exists(f"saved_model/gan/{data_name}/D.pth"):
    discriminator.load_state_dict(torch.load(
        f"saved_model/gan/{data_name}/D.pth"))


class RandomRotation90(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img):
        degree = torch.randint(0, 4, (1,)).item()
        ret = torch.rot90(img, degree, dims=[1, 2])
        return ret


# mean = [0.485, 0.456, 0.406]  # NOTE ImageNet mean, std
# std = [0.229, 0.224, 0.225]  # NOTE ImageNet mean, std
mean = [0.5, 0.5, 0.5]  # NOTE ImageNet mean, std
std = [0.5, 0.5, 0.5]  # NOTE ImageNet mean, std
transform1 = transforms.Compose([transforms.Resize(opt.img_size),])
transform2 = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.RandomVerticalFlip(),
     transforms.ToTensor(),
     # RandomRotation90(),
     transforms.Normalize(mean, std)]
)


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, paths):
        self.paths = []
        for path in paths:
            self.paths += [os.path.join(path, jpg_path)
                           for jpg_path in sorted(os.listdir(path))]
        self.get_transformed()  # NOTE resize 만이라도 미리 만들고 사용하자 매우 빨라짐

    def get_transformed(self):
        self.gan_data = []
        for path in tqdm.tqdm(self.paths, desc="Loading GANs data"):
            img = PIL.Image.open(path)
            self.gan_data.append(transform1(img))

    def __getitem__(self, index):
        return transform2(self.gan_data[index]), 0

    def __len__(self):
        return len(self.paths)


train_dataset = CustomDataset(
    data_config[data_name])
dataloader = torch.utils.data.DataLoader(
    # NOTE 왜 num_workers=1 보다 0이 빠르지?
    train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0
)

# Optimizers
optimizer_G = torch.optim.Adam(
    generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(
    discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

with tqdm.tqdm(range(opt.n_epochs)) as pbar:
    while pbar.n < opt.n_epochs:
        for i, (imgs, _) in enumerate(dataloader):

            # Adversarial ground truths
            valid = Variable(Tensor(imgs.shape[0], 1).fill_(
                1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.shape[0], 1).fill_(
                0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(
                0, 1, (imgs.shape[0], opt.latent_dim))))

            # Generate a batch of images
            gen_imgs = generator(z)[0]

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(
                discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            if pbar.n % opt.sample_interval == 0:
                save_path = f"saved_model/gan/{data_name}/gans"
                os.makedirs(save_path, exist_ok=True)
                save_real_path = f"saved_model/gan/{data_name}/reals"
                os.makedirs(save_real_path, exist_ok=True)
                save_image(gen_imgs.data[:25], os.path.join(
                    save_path, f"{pbar.n}.png"), nrow=5, normalize=True)
                save_image(imgs.data[:25], os.path.join(
                    save_real_path, f"{pbar.n}.png"), nrow=5, normalize=True)
            pbar.set_postfix(d_loss=d_loss.item(), g_loss=g_loss.item())
            pbar.update(1)

# Save model
torch.save(generator.state_dict(),
           f"saved_model/gan/{data_name}/G.pth")
torch.save(discriminator.state_dict(),
           f"saved_model/gan/{data_name}/D.pth")
