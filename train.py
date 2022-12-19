import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import random

from model import Discriminator, Generator, weights_init
from data_loader import celeba_loader, afhq_loader

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
    'nepochs': 5,  # Number of training epochs.
    'lr': 0.0002,  # Learning rate for optimizers
    'beta1': 0.5,  # Beta1 hyperparam for Adam optimizer
    'save_epoch': 2}  # Save step.


device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
dataloader = celeba_loader(params)
#dataloader = afhq_loader(params)

# Initialize the generator
gen_model = Generator(
    latent_size=params['nz'], n_channels=params['nc'], features_g=params['ngf']).to(device)
gen_model.apply(weights_init)
print(gen_model)

# Initialize the dsicriminator
disc_model = Discriminator(
    n_channels=params['nc'], features_d=params['ndf']).to(device)
disc_model.apply(weights_init)
print(disc_model)

# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, params['nz'], 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerG = optim.Adam(gen_model.parameters(),
                        lr=params['lr'], betas=(params['beta1'], 0.999))
optimizerD = optim.Adam(disc_model.parameters(),
                        lr=params['lr'], betas=(params['beta1'], 0.999))

# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(params['nepochs']):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # Train with all-real batch
        disc_model.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label,
                           dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = disc_model(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        # Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, params['nz'], 1, 1, device=device)
        # Generate fake image batch with G
        fake = gen_model(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = disc_model(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        gen_model.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = disc_model(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, params['nepochs'], i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == params['nepochs']-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = gen_model(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1

    # Save the model.
    if epoch % params['save_epoch'] == 0:
        torch.save({
            'generator': gen_model.state_dict(),
            'discriminator': disc_model.state_dict(),
            'optimizerG': optimizerG.state_dict(),
            'optimizerD': optimizerD.state_dict(),
            'params': params
        }, 'saved_models/model_epoch_{}.pth'.format(epoch))

# Save the final trained model.
torch.save({
    'generator': gen_model.state_dict(),
    'discriminator': disc_model.state_dict(),
    'optimizerG': optimizerG.state_dict(),
    'optimizerD': optimizerD.state_dict(),
    'params': params
}, 'saved_models/model_final.pth')


# Plot the training losses.
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
# plt.show()
plt.savefig('losses.png')

fig = plt.figure(figsize=(8, 8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)]
       for i in img_list]
ani = animation.ArtistAnimation(
    fig, ims, interval=1000, repeat_delay=1000, blit=True)
ani.save('generated.gif')
# HTML(ani.to_jshtml())
