#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utilities


Karen Ullrich, Oct 2017
"""

import os
import numpy as np
import imageio

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
cmap = sns.diverging_palette(240, 10, sep=100, as_cmap=True)

# -------------------------------------------------------
# VISUALISATION TOOLS
# -------------------------------------------------------

# Modified by Salman Siddique Khan
# This code has been modified from the original version to account for all layer visualization
# and saving of both random and pretrained initialized network's plot
def visualize_pixel_importance(imgs, log_alpha,FLAGS, epoch="pixel_importance"):
    num_imgs = len(imgs)

    f, ax = plt.subplots(1, num_imgs)
    plt.title("Epoch:" + epoch)
    for i, img in enumerate(imgs):
        img = (img / 255.) - 0.5
        mask = log_alpha.reshape(img.shape)
        mask = 1 - np.clip(np.exp(mask), 0.0, 1)
        ax[i].imshow(img * mask, cmap=cmap, interpolation='none', vmin=-0.5, vmax=0.5)
        ax[i].grid("off")
        ax[i].set_yticks([])
        ax[i].set_xticks([])
    if FLAGS.load_pretrained:
        plt.savefig("./.pretrained_pixel" + epoch + ".png", bbox_inches='tight')
    else:
        plt.savefig("./.pixel" + epoch + ".png", bbox_inches='tight')
    plt.close()


def visualise_weights(weight_mus, log_alphas, epoch,FLAGS):
     num_layers = len(weight_mus)

    for i in range(num_layers):
        f, ax = plt.subplots(1, 1)
        weight_mu = (weight_mus[i].cpu().data.numpy())
        # alpha
        log_alpha_fc1 = log_alphas[i].unsqueeze(0).cpu().data.numpy()
        log_alpha_fc1 = log_alpha_fc1 < thresholds[i]
        try:
            log_alpha_fc2 = log_alphas[i + 1].unsqueeze(1).cpu().data.numpy()
            log_alpha_fc2 = log_alpha_fc2 < thresholds[i+1]
        except:
            log_alpha_fc2 = np.ones((10,1))
        mask = log_alpha_fc1 * log_alpha_fc2
        num_nz = (mask>0).sum()
        tot_num = mask.shape[0] * mask.shape[1]
        ratio_z = 1-(num_nz/tot_num)
        # weight
        c = np.max(np.abs(weight_mu))
        s = ax.imshow(weight_mu * mask, cmap='seismic', interpolation='none', vmin=-c, vmax=c)
        ax.grid("off")
        ax.set_yticks([])
        ax.set_xticks([])
        s.set_clim([-c * 0.5, c * 0.5])
        f.colorbar(s)
        plt.title("Epoch:" + str(epoch))
        if not FLAGS.load_pretrained:
            plt.savefig("./.weight" + str(i) + '_e' + str(epoch) + ".png", bbox_inches='tight')
        else:
            plt.savefig("./.pretrained_weight" + str(i) + '_e' + str(epoch) + ".png", bbox_inches='tight')
        plt.close()


def generate_gif(save='tmp', epochs=10):
    images = []
    filenames = ["./." + save + "%d.png" % (epoch + 1) for epoch in np.arange(epochs)]
    for filename in filenames:
        images.append(imageio.imread(filename))
        #os.remove(filename)
    imageio.mimsave('./figures/' + save + '.gif', images, duration=.5)
