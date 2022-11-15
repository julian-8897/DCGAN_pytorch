import torch
import torchvision.transforms as transforms
from torchvision import datasets

# Directory containing the data.
root = 'data'


def celeba_loader(params):
    """
    Loads the dataset and applies proproccesing steps to it.
    Returns a PyTorch DataLoader.
    """
    # Data proprecessing.
    transform = transforms.Compose([
        transforms.Resize(params['imsize']),
        transforms.CenterCrop(params['imsize']),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))])

    # Create the dataset.
    dataset = datasets.ImageFolder(root=root, transform=transform)

    # Create the dataloader.
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=params['bsize'],
                                             shuffle=True)

    return dataloader


root_afhq = 'data_afhq/afhq/train'


def afhq_loader(params):
    """
    Loads the dataset and applies proproccesing steps to it.
    Returns a PyTorch DataLoader.
    """
    # Data proprecessing.
    transform = transforms.Compose([
        transforms.Resize(params['imsize']),
        transforms.CenterCrop(params['imsize']),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))])

    # Create the dataset.
    dataset = datasets.ImageFolder(root=root_afhq, transform=transform)

    # Create the dataloader.
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=params['bsize'],
                                             shuffle=True)

    return dataloader
