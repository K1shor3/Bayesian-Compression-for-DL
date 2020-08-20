#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Linear Bayesian Model


Karen Ullrich, Christos Louizos, Oct 2017

This code a modified version of the code by Karen Ullrich and Christos Louizos.
Modified by Salman Siddique Khan
"""


# libraries
from __future__ import print_function
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import BayesianLayers
from compression import compute_compression_rate, compute_reduced_weights
from utils import visualize_pixel_importance, generate_gif, visualise_weights

N = 60000.  # number of data points in the training set


def main():
    # import data
    kwargs = {'num_workers': 2} if FLAGS.cuda else {}

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))
                       ])),
        batch_size=FLAGS.batchsize, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))
        ])),
        batch_size=FLAGS.batchsize, shuffle=False, **kwargs)

    # for later analysis we take some sample digits
    mask = 255. * (np.ones((1, 28, 28))); print(FLAGS.cuda)
    examples = train_loader.sampler.data_source.data[5:10].numpy()
    images = np.vstack([mask, examples]); print("We will start training")
    


    if not FLAGS.load_pretrained:
        print('Starting from scratch')
        fc1_w_init = None
        fc1_b_init = None
        fc2_w_init = None
        fc2_b_init = None
        fc3_w_init = None
        fc3_b_init = None
    else:
        print('Starting from a pretrained point')
        ckpt_pret = torch.load('mnist_nn.pt')
        fc1_w_init = ckpt_pret['fc1.weight'].numpy()
        fc1_b_init = ckpt_pret['fc1.bias'].numpy()
        fc2_w_init = ckpt_pret['fc2.weight'].numpy()
        fc2_b_init = ckpt_pret['fc2.bias'].numpy()
        fc3_w_init = ckpt_pret['fc3.weight'].numpy()
        fc3_b_init = ckpt_pret['fc3.bias'].numpy()
    # build a simple MLP
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            # activation
            self.relu = nn.ReLU()
            # layers
            self.fc1 = BayesianLayers.LinearGroupNJ(28 * 28, 300, clip_var=0.04, init_weight=fc1_w_init, init_bias=fc1_b_init, cuda=FLAGS.cuda)
            self.fc2 = BayesianLayers.LinearGroupNJ(300, 100,init_weight=fc2_w_init, init_bias=fc2_b_init, cuda=FLAGS.cuda)
            self.fc3 = BayesianLayers.LinearGroupNJ(100, 10,init_weight=fc3_w_init, init_bias=fc3_b_init, cuda=FLAGS.cuda)
            # layers including kl_divergence
            self.kl_list = [self.fc1, self.fc2, self.fc3]

        def forward(self, x):
            x = x.view(-1, 28 * 28)
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            return self.fc3(x)

        def get_masks(self,thresholds):
            weight_masks = []
            mask = None
            for i, (layer, threshold) in enumerate(zip(self.kl_list, thresholds)):
                # compute dropout mask
                if mask is None:
                    log_alpha = layer.get_log_dropout_rates().cpu().data.numpy()
                    mask = log_alpha < threshold
                else:
                    mask = np.copy(next_mask)
                try:
                    log_alpha = layers[i + 1].get_log_dropout_rates().cpu().data.numpy()
                    next_mask = log_alpha < thresholds[i + 1]
                except:
                    # must be the last mask
                    next_mask = np.ones(10)

                weight_mask = np.expand_dims(mask, axis=0) * np.expand_dims(next_mask, axis=1)
                weight_masks.append(weight_mask.astype(np.float))
            return weight_masks

        def kl_divergence(self):
            KLD = 0
            for layer in self.kl_list:
                KLD += layer.kl_divergence()
            return KLD

    # init model
    model = Net().cuda();print('Loaded model')
    if FLAGS.cuda:
        model.cuda()

    # init optimizer
    optimizer = optim.Adam(model.parameters()); print('Loaded optimizer')

    # we optimize the variational lower bound scaled by the number of data
    # points (so we can keep our intuitions about hyper-params such as the learning rate)
    discrimination_loss = nn.functional.cross_entropy

    def objective(output, target, kl_divergence):
        discrimination_error = discrimination_loss(output, target)
        variational_bound = discrimination_error + kl_divergence / N
        if FLAGS.cuda:
            variational_bound = variational_bound.cuda()
        return variational_bound

    def train(epoch):
        model.train(); print('Entering training block');iter_num=0
        for data, target in train_loader:
            print(iter_num)
            data, target = data.cuda(),target.cuda(); #import pdb; pdb.set_trace()
            optimizer.zero_grad()
            output = model(data)
            loss = objective(output, target, model.kl_divergence())
            loss.backward()
            optimizer.step();iter_num +=1
            # clip the variances after each step
            for layer in model.kl_list:
                layer.clip_variances()
        print('Epoch: {} \tTrain loss: {:.6f} \t'.format(
            epoch, loss.item()))

    def test():
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                if FLAGS.cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)
                test_loss += discrimination_loss(output, target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        test_loss /= len(test_loader.dataset)
        print('Test loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    # train the model and save some visualisations on the way
    for epoch in range(1, FLAGS.epochs + 1):
        print('Now we will train the epoch:'+str(epoch))
        train(epoch)
        test()
        # visualizations
        weight_mus = [model.fc1.weight_mu, model.fc2.weight_mu]; #import pdb; pdb.set_trace()
        log_alphas = [model.fc1.get_log_dropout_rates(), model.fc2.get_log_dropout_rates(),
                      model.fc3.get_log_dropout_rates()]
        visualise_weights(weight_mus, log_alphas, epoch=epoch,FLAGS=FLAGS)
        log_alpha = model.fc1.get_log_dropout_rates().cpu().data.numpy()
        visualize_pixel_importance(images, log_alpha=log_alpha, FLAGS=FLAGS, epoch=str(epoch))
        if epoch%3 == 0:
            if not FLAGS.load_pretrained:
                torch.save(model.state_dict(), "epoch" + str(epoch) + "bcdl_no_pretrained.pt")
            else:
                torch.save(model.state_dict(), "epoch" + str(epoch) + "bcdl_pretrained.pt")
    if FLAGS.load_pretrained:
        generate_gif(save='pretrained_pixel', epochs=FLAGS.epochs)
        generate_gif(save='pretrained_weight0_e', epochs=FLAGS.epochs)
        generate_gif(save='pretrained_weight1_e', epochs=FLAGS.epochs)
    else:
        generate_gif(save='pixel', epochs=FLAGS.epochs)
        generate_gif(save='weight0_e', epochs=FLAGS.epochs)
        generate_gif(save='weight1_e', epochs=FLAGS.epochs)


    # compute compression rate and new model accuracy
    layers = [model.fc1, model.fc2, model.fc3]
    thresholds = FLAGS.thresholds
    compute_compression_rate(layers, model.get_masks(thresholds))

    print("Test error after with reduced bit precision:")

    weights = compute_reduced_weights(layers, model.get_masks(thresholds))
    for layer, weight in zip(layers, weights):
        if FLAGS.cuda:
            layer.post_weight_mu.data = torch.Tensor(weight).cuda()
        else:
            layer.post_weight_mu.data = torch.Tensor(weight)
    for layer in layers: layer.deterministic = True
    test()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--thresholds', type=float, nargs='*', default=[-2.8, -3., -5.])
    parser.add_argument('--load_pretrained', action='store_true', default=False)

    FLAGS = parser.parse_args()
    FLAGS.cuda = 1#torch.cuda.is_available()  # check if we can put the net on the GPU
    main()
