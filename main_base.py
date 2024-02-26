
'''
训练 base 模型
'''

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
from models.alexnet_pytorch import caffenet

import os
import click
import time
import numpy as np
from torchvision.transforms import ToPILImage
from torch.autograd import Variable
#from network.cifar10_net import wrn

from network import wideresnet
# import data_loader
from data import data_helper
import argparse
import pandas as pd

HOME = os.environ['HOME']

def get_args():
    parser = argparse.ArgumentParser(description="Script to launch jigsaw training",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source",  help="Source", nargs='+')
    parser.add_argument("--target",  help="Target")
    parser.add_argument("--batch_size", "-b", type=int, default=8, help="Batch size")
    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--min_scale", default=0.8, type=float, help="Minimum scale percent")
    parser.add_argument("--max_scale", default=1.0, type=float, help="Maximum scale percent")
    parser.add_argument("--random_horiz_flip", default=0.5, type=float, help="Chance of random horizontal flip")
    parser.add_argument("--jitter", default=0., type=float, help="Color jitter amount")
    parser.add_argument("--tile_random_grayscale", default=0.1, type=float, help="Chance of randomly greyscaling a tile")
    parser.add_argument("--limit_source", default=None, type=int,
                        help="If set, it will limit the number of training samples")
    parser.add_argument("--limit_target", default=None, type=int,
                        help="If set, it will limit the number of testing samples")
    parser.add_argument("--learning_rate", "-l", type=float, default=.002, help="Learning rate")
    parser.add_argument("--epochs", "-e", type=int, default=300, help="Number of epochs")
    parser.add_argument("--network", help="Which network to use", default="resnet18")
    parser.add_argument("--tf_logger", type=bool, default=True, help="If true will save tensorboard compatible logs")
    parser.add_argument("--val_size", type=float, default="0.1", help="Validation size (between 0 and 1)")
    parser.add_argument("--folder_name", default='test', help="Used by the logger to save logs")
    parser.add_argument("--bias_whole_image", default=0.9, type=float, help="If set, will bias the training procedure to show more often the whole image")
    parser.add_argument("--TTA", type=bool, default=False, help="Activate test time data augmentation")
    parser.add_argument("--classify_only_sane", default=False, type=bool, help="If true, the network will only try to classify the non scrambled images")
    parser.add_argument("--train_all", default=True, type=bool, help="If true, all network weights will be trained")
    parser.add_argument("--suffix", default="", help="Suffix for the logger")
    parser.add_argument("--nesterov", default=False, type=bool, help="Use nesterov")
    parser.add_argument("--visualization", default=False, type=bool)
    parser.add_argument("--epochs_min", type=int, default=1,
                        help="")
    parser.add_argument("--eval", default=False, type=bool)
    parser.add_argument("--ckpt", default="logs/model", type=str)
    parser.add_argument("--alpha1", default=1, type=float)
    parser.add_argument("--alpha2", default=1, type=float)
    parser.add_argument("--beta", default=0.1, type=float)
    parser.add_argument("--lr_sc", default=0.005, type=float)
    parser.add_argument("--task", default='PACS', type=str)

    return parser.parse_args()

@click.command()
@click.option('--gpu', type=str, default='0', help='选择gpu')
@click.option('--data', type=str, default='PACS', help='数据集名称')
@click.option('--ntr', type=int, default=None, help='选择训练集前ntr个样本')
@click.option('--translate', type=float, default=None, help='随机平移数据增强')
@click.option('--autoaug', type=str, default=None, help='AA FastAA RA')
@click.option('--epochs', type=int, default=300)
@click.option('--nbatch', type=int, default=None, help='每个epoch中batch的数量')
@click.option('--batchsize', type=int, default=128, help='每个batch中样本的数量')
@click.option('--lr', type=float, default=0.1)
@click.option('--lr_scheduler', type=str, default='none', help='是否选择学习率衰减策略')
@click.option('--svroot', type=str, default='./saved', help='项目文件保存路径')
def experiment(gpu, data, ntr, translate, autoaug, epochs, nbatch, batchsize, lr, lr_scheduler, svroot):
    settings = locals().copy()
    print(settings)

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    if not os.path.exists(svroot):
        os.makedirs(svroot)
    writer = SummaryWriter(svroot)

    if data in ['mnist', 'mnist_t']:

        if data == 'mnist':
            trset = data_loader.load_mnist('train', translate=translate, ntr=ntr, autoaug=autoaug)
        elif data == 'mnist_t':
            trset = data_loader.load_mnist_t('train', translate=translate, ntr=ntr)
        teset = data_loader.load_mnist('test')
        trloader = DataLoader(trset, batch_size=batchsize, num_workers=8, \
                sampler=RandomSampler(trset, True, nbatch*batchsize))
        teloader = DataLoader(teset, batch_size=batchsize, num_workers=8, shuffle=False)
        cls_net = mnist_net.ConvNet().cuda()
        cls_opt = optim.Adam(cls_net.parameters(), lr=lr)
    
    elif data == 'mnistvis':
        trset = data_loader.load_mnist('train')
        teset = data_loader.load_mnist('test')
        trloader = DataLoader(trset, batch_size=batchsize, num_workers=8, \
                sampler=RandomSampler(trset, True, nbatch*batchsize))
        teloader = DataLoader(teset, batch_size=batchsize, num_workers=8, shuffle=False)
        cls_net= mnist_net.ConvNetVis().cuda()
        cls_opt = optim.Adam(cls_net.parameters(), lr=lr)
    
    elif data == 'cifar10':

        trset = data_loader.load_cifar10(split='train')
        teset = data_loader.load_cifar10(split='test')
        trloader = DataLoader(trset, batch_size=batchsize, num_workers=8, shuffle=True, drop_last=True)
        teloader = DataLoader(teset, batch_size=batchsize, num_workers=8, shuffle=False)
        cls_net = wrn(depth=16, num_classes=10, widen_factor=4, dropRate=0.4, nc =3).cuda()
        cls_opt = optim.SGD(cls_net.parameters(), lr=smooth_step(10,40,100,150,0), momentum=0.9, weight_decay=1e-5)
        if lr_scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(cls_opt, epochs)
    elif data == 'PACS':
        args = get_args()
        args.n_classes = 7
        args.source = ['photo']
        args.target = ['art_painting', 'cartoon', 'sketch']
        trloader, teloader = data_helper.get_train_dataloader(args, patches=False)
        cls_net = caffenet(7).cuda()
        cls_opt = torch.optim.SGD(cls_net.parameters(), lr=0.002, nesterov=True, momentum=0.9, weight_decay=0.0005)
        scheduler = torch.optim.lr_scheduler.StepLR(cls_opt, step_size=int(epochs * 0.8))

    elif 'synthia' in data:

        branch = data.split('_')[1]
        trset = data_loader.load_synthia(branch)
        trloader = DataLoader(trset, batch_size=batchsize, num_workers=8, shuffle=True)
        teloader = DataLoader(trset, batch_size=batchsize, num_workers=8, shuffle=True)
        imsize = [192, 320]
        nclass = 14

        cls_net = fcn.FCN_resnet50(nclass=nclass).cuda()
        cls_opt = optim.Adam(cls_net.parameters(), lr=lr)
        if lr_scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(cls_opt, epochs*len(trloader))
    
    cls_criterion = nn.CrossEntropyLoss()

    # 开始训练
    best_acc = 0
    for epoch in range(epochs):
        t1 = time.time()
        
        loss_list = []
        cls_net.train()
        for i, ((x, _, y), _, idx) in enumerate(trloader):
            x = x.type(torch.FloatTensor)
            x, y = x.cuda(), y.cuda()

            p = cls_net(x)
            cls_loss = cls_criterion(p, y)
            #torch.cuda.synchronize()
            cls_opt.zero_grad()
            cls_loss.backward()
            cls_opt.step()
            
            loss_list.append([cls_loss.item()])

            scheduler.step()

        cls_loss, = np.mean(loss_list, 0)
        

        cls_net.eval()
        if data in ['mnist', 'mnist_t', 'cifar10', 'mnistvis']:
            teacc = evaluate_cifar10(cls_net, teloader)
        elif 'synthia' in data:
            teacc = evaluate_seg(cls_net, teloader, nclass)
        if data == 'PACS':
            teacc = evaluate_P(cls_net, teloader)

        if best_acc < teacc:
            best_acc = teacc
            torch.save({'cls_net':cls_net.state_dict()}, os.path.join(svroot, 'best.pkl'))

        # 保存日志
        t2 = time.time()
        print(f'epoch {epoch}, time {t2-t1:.2f}, cls_loss {cls_loss:.4f} teacc {teacc:2.2f}')
        writer.add_scalar('scalar/cls_loss', cls_loss, epoch)
        writer.add_scalar('scalar/teacc', teacc, epoch)

    writer.close()

def evaluate(net, teloader):
    correct, count = 0, 0
    ps = []
    ys = []
    for i,(x1, y1) in enumerate(teloader):
        with torch.no_grad():
            x1 = x1.cuda()
            p1 = net(x1)
            p1 = p1.argmax(dim=1)
            ps.append(p1.detach().cpu().numpy())
            ys.append(y1.numpy())

    ps = np.concatenate(ps)
    ys = np.concatenate(ys)
    acc = np.mean(ys==ps)*100
    return acc

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

def evaluate_P(model, test_loader, device='cuda'):
    model.eval()
    class_correct = 0
    total = 0
    for it, ((data, nouse, class_l), _, _) in enumerate(test_loader):
        data, nouse, class_l = data.to(device), nouse.to(device), class_l.to(device)
        z = model(data)
        _, cls_pred = z.max(dim=1)
        total += data.size(0)
        class_correct += torch.sum(cls_pred == class_l.data)

    acc = 100 * class_correct / total
    return acc

def evaluate_ACS(gpu, modelpath,data_loader,svpath='acs.test', device='cuda'):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    cls_net = caffenet(7).cuda()
    saved_weight = torch.load(modelpath, map_location='cuda')
    cls_net.load_state_dict(saved_weight['state'])
    cls_net.eval()


    columns = ['A', 'C', 'S']
    total = len(data_loader)
    avg_acc=0
    rst = []
    for loader in data_loader:
        teacc = evaluate_P(cls_net, loader)
        avg_acc += teacc
        rst.append(teacc.item())

    df = pd.DataFrame([rst], columns=columns)
    print(df)
    if svpath is not None:
        df.to_csv(svpath)

    acc = 100 * avg_acc / total
    print(acc.item())

    def evaluate_ACS(gpu, modelpath, data_loader, svpath='acs.test', device='cuda'):
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu

        cls_net = caffenet(7).cuda()
        saved_weight = torch.load(modelpath, map_location='cuda')
        cls_net.load_state_dict(saved_weight['state'])
        cls_net.eval()

        columns = ['A', 'C', 'S']
        total = len(data_loader)
        avg_acc = 0
        rst = []
        for loader in data_loader:
            teacc = evaluate_P(cls_net, loader)
            avg_acc += teacc
            rst.append(teacc.item())

        df = pd.DataFrame([rst], columns=columns)
        print(df)
        if svpath is not None:
            df.to_csv(svpath)

        acc = avg_acc / total
        print(acc.item())

def evaluate_ACS0(gpu, modelpath,data_loader,svpath='acs.test', device='cuda'):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    cls_net = caffenet(7).cuda()
    state_dict = torch.load(modelpath, map_location='cuda')
    state_dict["cls_head.weight"] = state_dict["classifier.6.weight"]
    state_dict["cls_head.bias"] = state_dict["classifier.6.bias"]
    del state_dict["classifier.6.weight"]
    del state_dict["classifier.6.bias"]
    # for key in state_dict:
    #     print(key)
    cls_net.load_state_dict(state_dict, strict=False)
    cls_net.eval()


    columns = ['A', 'C', 'S']
    total = len(data_loader)
    avg_acc=0
    rst = []
    for loader in data_loader:
        teacc = evaluate_P(cls_net, loader)
        avg_acc += teacc
        rst.append(teacc.item())

    df = pd.DataFrame([rst], columns=columns)
    print(df)
    if svpath is not None:
        df.to_csv(svpath)

    acc = avg_acc / total
    print(acc.item())

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("---  new folder...  ---")
        print("---  OK  ---")
    else:
        print("---  There is this folder!  ---")

def saveModel(model,epoch,test_acc,train_loss,path):
    torch.save(model, path+'/model.pkl')
    # torch.save(model.state_dict(), 'resnet.ckpt')
    epoch_save=np.array(epoch)
    np.save(path+'/learning_rate.npy',epoch_save)
    test_acc=np.array(test_acc)
    np.save(path+'/test_acc.npy',test_acc)
    train_loss=np.array(train_loss)
    np.save(path+'/train_loss.npy',train_loss)

def smooth_step(a, b, c, d, x):
    level_s = 0.01
    level_m = 0.1
    level_n = 0.01
    level_r = 0.005
    if x <= a:
        return level_s
    if a < x <= b:
        return (((x - a) / (b - a)) * (level_m - level_s) + level_s)
    if b < x <= c:
        return level_m
    if c < x <= d:
        return level_n
    if d < x:
        return level_r

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__=='__main__':
    # experiment()
    evaluate_ACS("0", "./saved/pytorch2/0-best.pkl",  './saved/pytorch2/0-best.pkl.test')

