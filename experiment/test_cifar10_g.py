import os
import torchvision
import torchvision.transforms as T
from torchvision import utils
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer


def get_dataset(name):
    if name == "cifar-10":
        if not os.path.exists('../data/cifar-10'):
            os.makedirs('../data/cifar-10')
            dataset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=T.ToTensor())
            for i in range(len(dataset)):
                img, _ = dataset[i]
                utils.save_image([img], f'../data/cifar-10/img-{i+1}.png')
    else:
        raise NotImplementedError


def train_gaussian():
    model = Unet(
        dim=128,
        dim_mults=(1, 2, 2, 2)
    )

    diffusion = GaussianDiffusion(
        model,
        image_size=32,
        timesteps=1000,   # number of steps
        loss_type='l2'    # L1 or L2
    )

    get_dataset('cifar-10')
    trainer = Trainer(
        diffusion,
        '../data/cifar-10',
        train_batch_size=128,
        train_lr=2e-4,
        train_num_steps=70000,  # total training steps
        save_and_sample_every=2000,
        gradient_accumulate_every=1,  # gradient accumulation steps
        ema_decay=0.9999,  # exponential moving average decay
        amp=False,  # turn on mixed precision
        results_folder='./g_results'
    )
    trainer.train()


if __name__ == "__main__":
    train_gaussian()
