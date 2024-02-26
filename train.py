import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, RandomSampler
from torchvision import models
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import random
import os
import click
import time
import numpy as np
from utils import *
from itertools import chain
from network import mnist_net, generator, wideresnet
import data_loader
from evalue_loader import evaluate, evaluate_digit_save, evaluate_cifar10

HOME = os.environ['HOME']
@click.command()
@click.option('--gpu', type=str, default='0', help='select gpu ')
@click.option('--data', type=str, default='mnist', help='dataset name')
@click.option('--ntr', type=int, default=None, help='select the first ntr samples of the training set')
@click.option('--n_tgt', type=int, default=5, help='number of tgt model')
@click.option('--tgt_epochs', type=int, default=160, help='number of epochs trained per target domain')
@click.option('--tgt_epochs_fixg', type=int, default=150, help='G_fixed thresholds')
@click.option('--nbatch', type=int, default=100, help='number of batch in each epoch')
@click.option('--batchsize', type=int, default=256)
@click.option('--lr', type=float, default=1e-3)
@click.option('--lr_scheduler', type=str, default='none', help='learning rate decay strategy')
@click.option('--svroot', type=str, default='./saved')
@click.option('--ckpt', type=str, default='./saved/best.pkl')
@click.option('--n_net', type=float, default=16.0, help='network layer')
@click.option('--w_noise', type=float, default=0.2, help='weights for noise')
@click.option('--dir', type=str, default=None, help='storage directory')
@click.option('--eval', type=bool, default=False, help='evaluation')
def experiment(gpu, data, ntr, n_tgt, tgt_epochs, tgt_epochs_fixg, nbatch, batchsize, lr,
               lr_scheduler, svroot, ckpt, n_net, w_noise, dir, eval):
    settings = locals().copy()
    print(settings)

    output_dir = make_output_dir(dir)

    model_dir = os.path.join(output_dir, 'Model')
    os.makedirs(model_dir)
    image_dir = os.path.join(output_dir, 'Image')
    os.makedirs(image_dir)
    csv_dir = os.path.join(output_dir, 'Record')
    os.makedirs(csv_dir)

    file_n = os.path.join(csv_dir, f'{data}.csv')
    file_n_best = os.path.join(csv_dir, f'{data}_best.csv')

    ####  Evaluation ####
    if eval is True:
        model_path =  './models_pth/SRC_net_digits.pth'
        rst = evaluate_digit_save(gpu, model_path, file_n, batchsize=batchsize)
        print('Comparison result (in %) on digits')

    zdim = 10
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    g1root = os.path.join(svroot, 'g1')
    if not os.path.exists(g1root):
        os.makedirs(g1root)
    writer = SummaryWriter(svroot)

    # datasets
    imdim = 3
    image_size = (32, 32)
    preprocess = transforms.Compose([
        transforms.Resize(image_size, antialias=True),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((32, 32), (0.8, 1.0), antialias=True),
        transforms.Resize(image_size, antialias=True),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    test_transform = preprocess
    print("\n=========Building Model=========")
    print(f'train_transform {train_transform}')
    print(f'test_transform {test_transform}')

    if data in ['mnist', 'mnistvis']:
        trset = data_loader.load_mnist('train', ntr=ntr, translate=train_transform)
        teset = data_loader.load_mnist('test', ntr=ntr, translate=test_transform)
    imsize = [32, 32]

    trloader = DataLoader(trset, batch_size=batchsize, num_workers=8,
                          sampler=RandomSampler(trset, True, nbatch * batchsize))
    teloader = DataLoader(teset, batch_size=batchsize, num_workers=8, shuffle=False)

    def get_generator(name):
        if name == 'cnn':
            g1_net = generator.Generation_G(n=16, w_noise=0.2, imdim=imdim, imsize=imsize).cuda()
            g2_net = generator.Generation_Phi(imdim=imdim, imsize=imsize).cuda()
            g1_opt = optim.Adam(g1_net.parameters(), lr=lr)
            g2_opt = optim.Adam(g2_net.parameters(), lr=lr)
        return g1_net, g2_net, g1_opt, g2_opt

    g1_list = []
    if data in ['mnist', 'mnist_t']:
        src_net = mnist_net.ConvNet().cuda()
        # saved_weight = torch.load(ckpt)
        # src_net.load_state_dict(saved_weight['cls_net'])
        src_opt = optim.Adam(src_net.parameters(), lr=lr)

    elif data == 'mnistvis':
        src_net = mnist_net.ConvNetVis().cuda()
        saved_weight = torch.load(ckpt)
        src_net.load_state_dict(saved_weight['cls_net'])
        src_opt = optim.Adam(src_net.parameters(), lr=lr)

    cls_criterion = nn.CrossEntropyLoss()

    # trainning
    global_best_acc = 0
    for i_tgt in range(n_tgt):
        print(f'target domain {i_tgt}')
        if lr_scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(src_opt, tgt_epochs * len(trloader))
        g1_net, g2_net, g1_opt, g2_opt = get_generator(gen)
        best_acc = 0
        for epoch in range(tgt_epochs):
            t1 = time.time()

            flag_fixG = False
            if (tgt_epochs_fixg is not None) and (epoch >= tgt_epochs_fixg):
                flag_fixG = True
            loss_list = []
            time_list = []
            src_net.train()
            for i, (x, y) in enumerate(trloader):
                x, y = x.cuda(), y.cuda()

                rand = 0.01
                x_probe = g1_net(x)
                x_trace = g2_net(x)
                x3_mix = rand * x + (1. - rand) * x_probe

                p1_src, z1_src = src_net(x, mode='train')
                p_tgt, z_tgt = src_net(x3_mix.detach(), mode='train')

                src_cls_loss = cls_criterion(p1_src, y)
                tgt_cls_loss = cls_criterion(p_tgt, y)

                loss = src_cls_loss + tgt_cls_loss

                src_opt.zero_grad()
                loss.backward()
                src_opt.step()

                if flag_fixG:
                    # fix G，train task_net
                    loss_G = torch.tensor(0)
                    loss_Gadv = torch.tensor(0)
                else:
                    ### Train Auxiliary_Phi ####

                    #### Train Generater_G ####

                # update
                if lr_scheduler in ['cosine']:
                    scheduler.step()

            # test
            src_net.eval()
            teacc = evaluate(src_net, teloader)
            torch.save(src_net.state_dict(), '%s/SRC_net_%d.pth' % (model_dir, epoch))
            pklpath = f'{model_dir}/SRC_net_{epoch}.pth'
            t2 = time.time()
            print(f'target domain {i_tgt} Epoch: %d, Fished: time {t2 - t1:.2f}' % (epoch))
            rst = evaluate_digit_save(gpu, pklpath, file_n, batchsize=batchsize)
            if rst[-1] >= best_acc:
                best_acc = rst[-1]
                columns = ['mnist', 'svhn', 'mnist_m', 'syndigit', 'usps', 'ave']
                df = pd.DataFrame([rst], columns=columns).to_csv(file_n_best, index=False, mode='a+', header=False)
                torch.save(g1_net.state_dict(), '%s/G1_%d.pth' % (model_dir, epoch))
                torch.save(g2_net.state_dict(), '%s/G2_%d.pth' % (model_dir, epoch))

            l_list = []
            l_list.append(make_grid(x[0:10].detach().cpu(), 1, 2, pad_value=128))
            l_list.append(make_grid(x_probe[0:10].detach().cpu(), 1, 2, pad_value=128))
            l_list.append(make_grid(x_trace[0:10].detach().cpu(), 1, 2, pad_value=128))
            l_list.append(make_grid(x3_mix[0:10].detach().cpu(), 1, 2, pad_value=128))
            rst = make_grid(torch.stack(l_list), len(l_list), pad_value=128)
            PIL_img = transforms.ToPILImage()(rst.float())
            file_name = f'{image_dir}/domain_{i_tgt}_im_gen_{epoch}+{i}.png'
            PIL_img.save(file_name)

            src_net.train()

def pearson_correlation(x, y):
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    x_trace = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = torch.sum(x_trace * ym)
    r_den = torch.sqrt(torch.sum(x_trace ** 2) * torch.sum(ym ** 2))
    r = r_num / r_den
    return r

if __name__ == '__main__':
    manualSeed = 0 #random.randint(1, 10000)
    print(f' manualSeed is {manualSeed}')
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.deterministic = True
    experiment()

