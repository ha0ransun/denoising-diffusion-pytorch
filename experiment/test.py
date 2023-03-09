import torchvision.transforms as T
import torchvision
import torch
import math
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import numpy as np


def get_weight(x, sigma):
    d = torch.cdist(x, x) ** 2
    logit = d * math.sqrt(1 - sigma ** 2) / (2 * sigma ** 2 + 1e-9)
    return torch.softmax(logit, dim=-1)


def get_entropy(x, weight, sigma):
    res = x.shape[-1] * (math.log(sigma + 1e-9) + (math.log(math.pi) + 1) / 2.) + math.log(x.shape[0])
    mu = weight @ x
    res += ((1 - sigma ** 2) * (weight @ (x * x).sum(-1) - (mu * mu).sum(-1)) / (sigma ** 2 + 1e-9) - x.shape[-1]).mean()
    return torch.clamp(res, min=0.)


def get_entropy_lb(distance, sigma):
    distance = distance * (1 - sigma ** 2)
    h = torch.logsumexp(torch.clamp(- distance / (4 * sigma ** 2), min=-200.), -1).sum()
    return h


def get_all_entropy(x, num, batch_size):
    all_entropy = torch.zeros(num, device=x.device)
    for i in tqdm(range(int(np.ceil(x.shape[0] / batch_size)))):
        cur_x = x[i * batch_size: (i + 1) * batch_size]
        cur_d = torch.round((torch.cdist(cur_x, x) ** 2) * 4 / 255 ** 2, decimals=6)
        for j in range(num):
            sigma = (j + 1) / num
            all_entropy[j] += get_entropy_lb(cur_d, sigma).float()
    all_entropy /= x.shape[0]
    return all_entropy


def get_cov_entropy(x, num, batch_size):
    cov = torch.cov(x.T * 2 / 255) * 0.999 + 0.001 * torch.eye(x.shape[-1], device=x.device)
    cov /= (torch.logdet(cov) / 3072).exp()
    v, P = torch.linalg.eigh(cov)
    W = P @ torch.diag(v ** -0.5) @ P.T
    x_W = torch.round(x @ W, decimals=3)
    all_entropy = torch.zeros(num, device=x.device)
    for i in tqdm(range(int(np.ceil(x.shape[0] / batch_size)))):
        cur_x = x_W[i * batch_size: (i + 1) * batch_size]
        cur_d = torch.round((torch.cdist(cur_x, x_W) ** 2) * 4 / 255 ** 2, decimals=6)
        for j in range(num):
            sigma = (j + 1) / num
            all_entropy[j] += get_entropy_lb(cur_d, sigma).float()
    all_entropy /= x.shape[0]
    return all_entropy


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=T.ToTensor())
    x = torch.tensor(dataset.data).to(device)
    x = x.reshape(x.shape[0], -1)
    x = x.double()
    num = 1000
    res = get_all_entropy(x, num=num, batch_size=1000).cpu().numpy()
    with open('results/entropy_isotropic.pkl', 'wb') as file:
        pickle.dump(res, file)
    plt.plot(range(num), res)
    plt.savefig('results/entropy_isotropic.png')
    res = get_cov_entropy(x, num=num, batch_size=1000).cpu().numpy()
    with open('results/entropy_cov.pkl', 'wb') as file:
        pickle.dump(res, file)
    plt.plot(range(num), res)
    plt.savefig('results/entropy_cov.png')