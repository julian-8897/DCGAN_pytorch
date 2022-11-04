import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

from model import Discriminator, Generator, weights_init
from data_loader import celeba_loader

seed = 999
random.seed(seed)
torch.manual_seed(seed)
print("Random Seed: ", seed)

# Parameters to define the model.
params = {
    "bsize": 128,  # Batch size during training.
    # Spatial size of training images. All images will be resized to this size during preprocessing.
    'imsize': 64,
    # Number of channles in the training images. For coloured images this is 3.
    'nc': 3,
    'nz': 100,  # Size of the Z latent vector (the input to the generator).
    # Size of feature maps in the generator. The depth will be multiples of this.
    'ngf': 64,
    # Size of features maps in the discriminator. The depth will be multiples of this.
    'ndf': 64,
    'nepochs': 10,  # Number of training epochs.
    'lr': 0.0002,  # Learning rate for optimizers
    'beta1': 0.5,  # Beta1 hyperparam for Adam optimizer
    'save_epoch': 2}  # Save step.


device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

dataloader = celeba_loader(params)
