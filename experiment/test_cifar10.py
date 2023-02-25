import torch
import os
import torchvision
import torchvision.transforms as T
from torchvision import utils
from PIL import Image
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer, GPDiffusion


def get_dataset(name):
    if name == "cifar-10":
        if os.path.exists('../data/cifar-10'):
            dataset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=T.ToTensor())
        else:
            os.makedirs('../data/cifar-10')
            dataset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=T.ToTensor())
            for i in range(len(dataset)):
                img, _ = dataset[i]
                utils.save_image([img], f'../data/cifar-10/img-{i+1}.png')
    else:
        raise NotImplementedError
    kernel = torch.corrcoef(torch.stack([dataset[j][0].view(-1) for j in range(len(dataset))], 1))
    v, P = torch.linalg.eigh(kernel)
    return P @ torch.diag(v ** 0.5) @ P.T


def train_gaussian():
    model = Unet(
        dim=16,
        dim_mults=(1, 2, 4, 8)
    )

    diffusion = GaussianDiffusion(
        model,
        image_size=32,
        timesteps=10,   # number of steps
        loss_type='l1'    # L1 or L2
    )

    get_dataset('cifar-10')
    trainer = Trainer(
        diffusion,
        '../data/cifar-10',
        train_batch_size=16,
        train_lr=8e-5,
        train_num_steps=700,  # total training steps
        save_and_sample_every=100,
        gradient_accumulate_every=1,  # gradient accumulation steps
        ema_decay=0.995,  # exponential moving average decay
        amp=True,  # turn on mixed precision
        results_folder='./g_results'
    )
    trainer.train()


def train_gp():
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8)
    )

    kernel = get_dataset('cifar-10')
    diffusion = GPDiffusion(
        model,
        kernel=kernel,
        image_size=32,
        timesteps=1000,   # number of steps
        loss_type='l1'    # L1 or L2
    )

    trainer = Trainer(
        diffusion,
        '../data/cifar-10',
        train_batch_size=32,
        train_lr=8e-5,
        train_num_steps=70000,  # total training steps
        save_and_sample_every=1000,
        gradient_accumulate_every=1,  # gradient accumulation steps
        ema_decay=0.995,  # exponential moving average decay
        amp=True,  # turn on mixed precision
        results_folder='./gp_results'
    )
    trainer.train()


if __name__ == "__main__":
    # train_gaussian()
    train_gp()