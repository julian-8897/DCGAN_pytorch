<h1 align="center">
  <b>DCGAN in PyTorch</b><br>
</h1>

<p align="center">
      <a href="https://www.python.org/">
        <img src="https://img.shields.io/badge/Python-3.9-ff69b4.svg" /></a>
       <a href= "https://pytorch.org/">
        <img src="https://img.shields.io/badge/PyTorch-1.9-2BAF2B.svg" /></a>
       <a href= "https://github.com/julian-8897/Vanilla-VAE-PyTorch/blob/master/LICENSE.md">
        <img src="https://img.shields.io/badge/license-MIT-blue.svg" /></a>
         
</p>

## Implementation Details

A Pytorch implementation of the paper: "[Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)".

This implementation supports model training on the [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). This project serves as a proof of concept, hence the original images (178 x 218) are scaled and cropped to (64 x 64) images in order to speed up the training process. For ease of access, the zip file which contains the dataset can be downloaded from: https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip.

## Installation Guide

```
$ git clone https://github.com/julian-8897/DCGAN_pytorch
$ cd DCGAN_pytorch
$ pip install -r requirements.txt
```

## Usage

### Training

To train the model, edit the hyperparameters in 'train.py', and run the following command:

```
python train.py
```

### Model Evaluation

To evaluate the model and generate samples, run the following command:

```
python test.py
```

<h2 align="left">
  <b>Results</b><br>
</h2>

#### learning rate = 0.0002, batch size = 128, trained for 10 epochs with random seed

| Generated Samples |
| ----------------- |
| ![][1]            |

| Linear Interpolation Between Two Images |
| --------------------------------------- |
| ![][4]                                  |

| Loss Plots |
| ---------- |
| ![][2]     |

| Generator's Output after each training epoch |
| -------------------------------------------- |
| ![][3]                                       |

[1]: https://github.com/julian-8897/DCGAN_pytorch/blob/main/results/dcgan_generated_epoch_10.png
[2]: https://github.com/julian-8897/DCGAN_pytorch/blob/main/results/dcgan_losses_epoch_10.png
[3]: https://github.com/julian-8897/DCGAN_pytorch/blob/main/results/generated_epoch_10.gif
[4]: https://github.com/julian-8897/DCGAN_pytorch/blob/main/results/linear_interpolation.png
