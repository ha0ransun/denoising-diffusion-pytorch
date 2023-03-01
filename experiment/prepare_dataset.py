import torchvision
import os
import torchvision.transforms as T
from torchvision import utils


def get_cifar10():
    if not os.path.exists('../data/cifar-10'):
        os.makedirs('../data/cifar-10')
        dataset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=T.ToTensor())
        for i in range(len(dataset)):
            img, _ = dataset[i]
            utils.save_image([img], f'../data/cifar-10/img-{i+1}.png')


if __name__ == "__main__":
    get_cifar10()