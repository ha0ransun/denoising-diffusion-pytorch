import torch
import os
import torchvision
import torchvision.transforms as T
from torchvision import utils
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer


def get_kernel(name):
    if name == "cifar-10":
        dataset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=T.ToTensor())
        kernel = torch.corrcoef(torch.stack([dataset[j][0].view(-1) for j in range(len(dataset))], 1))
        kernel = (kernel + torch.diag(torch.diag(kernel))) / 2.
        # kernel = torch.cov(2 * torch.stack([dataset[j][0].view(-1) for j in range(len(dataset))], 1))
        v, P = torch.linalg.eigh(kernel)
        return P @ torch.diag(v ** 0.5) @ P.T, P @ torch.diag(v ** -1) @ P.T
    else:
        raise NotImplementedError


def train_gp():
    model = Unet(
        dim=128,
        dim_mults=(1, 2, 2, 2)
    )

    kernel, k_inv = get_kernel('cifar-10')
    diffusion = GaussianDiffusion(
        model,
        kernel=kernel,
        k_inv=k_inv,
        image_size=32,
        timesteps=1000,   # number of steps
        loss_type='K'    # L1 or L2 or K
    )

    trainer = Trainer(
        diffusion,
        '../data/cifar-10',
        train_batch_size=512,
        train_lr=2e-4,
        train_num_steps=200000,  # total training steps
        save_and_sample_every=2000,
        gradient_accumulate_every=1,  # gradient accumulation steps
        ema_decay=0.9999,  # exponential moving average decay
        amp=False,  # turn on mixed precision
        results_folder='./k_results'
    )
    trainer.train()


if __name__ == "__main__":
    train_gp()

