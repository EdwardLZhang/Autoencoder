import torch
import torchvision.datasets
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def loadCIFAR10() -> (DataLoader, DataLoader):
    # loads both the train and test datasets for CIFAR10
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    print('Downloading CIFAR10 training dataset')

    traindata = torchvision.datasets.CIFAR10(
        root='data/CIFAR10',
        train=True,
        transform=transform,
        download=True
    )

    print('Downloading CIFAR10 test dataset')

    testdata = torchvision.datasets.CIFAR10(
        root='data/CIFAR10',
        train=False,
        transform=transform,
        download=True
    )
    print('Loading CIFAR10 training dataset')
    trainloader = DataLoader(traindata, batch_size=64, shuffle=True)
    print('Loading CIFAR10 testing dataset')
    testloader = DataLoader(testdata, batch_size=64, shuffle=True)
    return trainloader, testloader


if __name__ == '__main__':
    loadCIFAR10()