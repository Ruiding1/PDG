import numpy as np
import os
import time
import torch.backends.cudnn as cudnn
import torch
import torchvision.utils as vutils
import sys
import shutil
import torch.nn as nn
import math
from torch.nn import functional as F
from torchvision import transforms
from torchvision import datasets
from network.wideresnet import WideResNet
from torch.utils.data import DataLoader
from network import mnist_net, generator, model_FID1
from lib.datasets.transforms import GreyToColor, IdentityTransform, ToGrayScale, LaplacianOfGaussianFiltering
import pandas as pd
from network.rand_conv import RandConvModule
import random
from PIL import Image
from torch.autograd import Variable

CORRUPTIONS_NOISE = [
    'gaussian_noise', 'shot_noise', 'speckle_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]# 哪个四类

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]# 哪个四类

def log(txt, path):
    print(txt)
    with open(path, 'a') as f:
        f.write(txt + '\n')
        f.flush()
        f.close()


def to_rad(deg):
    return deg/180*math.pi

def conditional_mmd_rbf(source, target, label, num_class, kernel_mul=2.0, kernel_num=5, fix_sigma=None, ver=2):
    loss = 0
    for i in range(num_class):
        source_i = source[label==i]
        target_i = target[label==i]
        loss += mmd_rbf(source_i, target_i)
    return loss / num_class


def cost_matrix(x, y):
    xy_T = torch.matmul(x, y.transpose(0, 1))
    x2 = torch.sum(torch.square(x), dim=1, keepdim=True)
    y2 = torch.sum(torch.square(y), dim=1, keepdim=True)
    norm = torch.matmul(torch.sqrt(x2), torch.sqrt(y2).transpose(0, 1))
    C = 1 - xy_T / norm
    return C


def sink(M, reg=1, numItermax=1000, stopThr=1e-9, cuda=True):

        # we assume that no distances are null except those of the diagonal of
        # distances

        if cuda:
            a = Variable(torch.ones((M.size()[0],)) / M.size()[0]).cuda()
            b = Variable(torch.ones((M.size()[1],)) / M.size()[1]).cuda()
        else:
            a = Variable(torch.ones((M.size()[0],)) / M.size()[0])
            b = Variable(torch.ones((M.size()[1],)) / M.size()[1])

        # init data
        Nini = len(a)
        Nfin = len(b)

        if cuda:
            u = Variable(torch.ones(Nini) / Nini).cuda()
            v = Variable(torch.ones(Nfin) / Nfin).cuda()
        else:
            u = Variable(torch.ones(Nini) / Nini)
            v = Variable(torch.ones(Nfin) / Nfin)

        # print(reg)

        K = torch.exp(-M / reg)
        # print(np.min(K))

        Kp = (1 / a).view(-1, 1) * K
        cpt = 0
        err = 1
        while (err > stopThr and cpt < numItermax):
            uprev = u
            vprev = v
            # print(T(K).size(), u.view(u.size()[0],1).size())
            KtransposeU = K.t().matmul(u)
            v = torch.div(b, KtransposeU)
            u = 1. / Kp.matmul(v)

            if cpt % 10 == 0:
                # we can speed up the process by checking for the error only all
                # the 10th iterations
                transp = u.view(-1, 1) * (K * v)
                err = (torch.sum(transp) - b).norm(1).pow(2).item()

            cpt += 1

        return torch.sum(u.view((-1, 1)) * K * v.view((1, -1)) * M)

def evaluate_cifar10(model, test_loader, device='cuda'):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        acc = 100 * correct / total

        # print('Accuracy of the model on the test images: {} %'.format(acc))
    return acc


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def calculate_correct(scores, labels):
    assert scores.size(0) == labels.size(0)
    _, pred = scores.max(dim=1)
    correct = torch.sum(pred.eq(labels)).item()
    return correct



class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin, alpha):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        cos_reg = cos(negative - anchor, positive - anchor).sum(0)
        losses = F.relu(distance_positive - distance_negative + self.margin - self.alpha * cos_reg)  # 2e-2

        return losses.mean()



def pairwise_distances(x, y):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)

def model_unet(model_input, in_channel=3, out_channel=1):
    model_test = model_input(in_channel, out_channel)
    return model_test


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)#/len(kernel_val)

def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None, ver=2):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

    loss = 0

    if ver == 1:
        for i in range(batch_size):
            s1, s2 = i, (i + 1) % batch_size
            t1, t2 = s1 + batch_size, s2 + batch_size
            loss += kernels[s1, s2] + kernels[t1, t2]
            loss -= kernels[s1, t2] + kernels[s2, t1]
        loss = loss.abs_() / float(batch_size)
    elif ver == 2:
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
    else:
        raise ValueError('ver == 1 or 2')

    return loss

def domain_mmd_rbf(source, target, num_domain, d_label, kernel_mul=2.0, kernel_num=5, fix_sigma=None, ver=2):
    loss = 0
    loss_overall = mmd_rbf(source, target)
    for i in range(num_domain):
        source_i = source[d_label == i]
        target_i = target[d_label == i]
        loss += mmd_rbf(source_i, target_i)
    return loss_overall - loss / num_domain

def domain_conditional_mmd_rbf(source, target, num_domain, d_label, num_class, c_label):
    loss = 0
    for i in range(num_class):
        source_i = source[c_label == i]
        target_i = target[c_label == i]
        d_label_i = d_label[c_label == i]
        loss_c = mmd_rbf(source_i, target_i)
        loss_d = 0
        for j in range(num_domain):
            source_ij = source_i[d_label_i == j]
            target_ij = target_i[d_label_i == j]
            loss_d += mmd_rbf(source_ij, target_ij)
        loss += loss_c - loss_d / num_domain

    return loss / num_class
def reparametrize(mu, logvar, factor=0.2):
    std = logvar.div(2).exp()
    eps = std.data.new(std.size()).normal_()
    return mu + factor*std*eps


def loglikeli(mu, logvar, y_samples):
    return (-(mu - y_samples)**2 /logvar.exp()-logvar).mean()#.sum(dim=1).mean(dim=0)

def club(mu, logvar, y_samples):

    sample_size = y_samples.shape[0]
    # random_index = torch.randint(sample_size, (sample_size,)).long()
    random_index = torch.randperm(sample_size).long()

    positive = - (mu - y_samples) ** 2 / logvar.exp()
    negative = - (mu - y_samples[random_index]) ** 2 / logvar.exp()
    upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
    return upper_bound / 2.

def get_random_module( data_mean, data_std):
    return RandConvModule(
                          in_channels=3,
                          out_channels= 3,
                          kernel_size = [1, 1],
                          mixing= False,
                          identity_prob=  0.5,
                          rand_bias=  False,
                          distribution= 'kaiming_normal',
                          data_mean=data_mean, # (0.5, 0.5, 0.5)
                          data_std=data_std, # data_std
                          clamp_output= False,
                          )

def make_output_dir(dir):
    root_path = './result'
    args = dir
    if len(args)==0:
        raise RuntimeError('output folder must be specified')
    new_output = args
    path = os.path.join(root_path, new_output)
    if os.path.exists(path):
        if len(args)==2 and args =='-f':
            print('WARNING: experiment directory exists, it has been erased and recreated')
            shutil.rmtree(path)
            os.makedirs(path)
        else:
            print('WARNING: experiment directory exists, it will be erased and recreated in 3s')
            time.sleep(3)
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)
    return path


def toggle_grad(model, on_or_off):
    for param in model.parameters():
        param.requires_grad = on_or_off
def make_output_dir(dir):
    root_path = './result'
    args = dir
    if len(args)==0:
        raise RuntimeError('output folder must be specified')
    new_output = args
    path = os.path.join(root_path, new_output)
    if os.path.exists(path):
        if len(args)==2 and args =='-f':
            print('WARNING: experiment directory exists, it has been erased and recreated')
            shutil.rmtree(path)
            os.makedirs(path)
        else:
            print('WARNING: experiment directory exists, it will be erased and recreated in 3s')
            time.sleep(3)
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)
    return path