import torch
import torchvision
import torchvision.datasets as datasets
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer, GaussianProcessDiffusion


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

    trainer = Trainer(
        diffusion,
        'cifar10',
        train_batch_size=16,
        train_lr=8e-5,
        train_num_steps=700,  # total training steps
        save_and_sample_every=100,
        gradient_accumulate_every=1,  # gradient accumulation steps
        ema_decay=0.995,  # exponential moving average decay
        amp=True  # turn on mixed precision
    )
    trainer.train()
    # after a lot of training

    sampled_images = diffusion.sample(batch_size=4)
    sampled_images.shape # (4, 3, 32, 32)


if __name__ == "__main__":
    train_gaussian()