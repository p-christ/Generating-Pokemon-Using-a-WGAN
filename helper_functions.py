import os
import tensorflow as tf
import scipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import imageio


def view_samples(epoch, samples, nrows, ncols, figsize=(5,5)):
    """Displays the images that come out of the network"""
    fig, axes = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols, 
                             sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples[epoch]):
        ax.axis('off')
        img = (img + 1.0)/2.0
        img = img*255.0
        ax.set_adjustable('box-forced')
        im = ax.imshow(img, aspect='equal')
   
    plt.subplots_adjust(wspace=0, hspace=0)
    return fig, axes

def showImagesHorizontally(list_of_files, height=15, width=15):
    """Show a list of images in a horizontal line.
    Args:
        list_of_files: a list of paths to png images
        height: height you want the images to be
        width: width you want the images to be"""

    fig = plt.figure()
    fig.set_figheight(height)
    fig.set_figwidth(width)
    number_of_files = len(list_of_files)
    for i in range(number_of_files):
        a=fig.add_subplot(1,number_of_files,i+1)
        image = imageio.imread(list_of_files[i])
        plt.imshow(image,cmap='Greys_r')
        plt.axis('off')

def lrelu(x):
    """Leaky Relu activation function which is sometimes preferred over
    Relu for GANs."""
    return tf.maximum(x, tf.multiply(x, 0.2))
