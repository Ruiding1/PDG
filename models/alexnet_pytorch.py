from collections import OrderedDict
import torch
import torch.nn as nn
from utils import *

import os
from itertools import chain

class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        print("Using Pytorch AlexNet")
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            # nn.Linear(4096, num_classes),
        )

        self.cls_head = nn.Linear(4096, num_classes)
        #
        self.pro_head = nn.Linear(4096, 128)

    def forward(self, x: torch.Tensor, mode='test'):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        if mode == 'test':
            p = self.cls_head(x)
            return p
        elif mode == 'train':
            p = self.cls_head(x)
            z = self.pro_head(x)
            z = F.normalize(z)
            return p, z
        return x
class AlexNetCaffe(nn.Module):
    def __init__(self, n_classes=100, dropout=True):
        super(AlexNetCaffe, self).__init__()
        print("Using Caffe AlexNet")
        self.features = nn.Sequential(OrderedDict([
            ("0", nn.Conv2d(3, 96, kernel_size=11, stride=4)),
            ("relu1", nn.ReLU(inplace=True)),
            ("pool1", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
            ("norm1", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
            ("4", nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)),
            ("relu2", nn.ReLU(inplace=True)),
            ("pool2", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
            ("norm2", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
            ("8", nn.Conv2d(256, 384, kernel_size=3, padding=1)),
            ("relu3", nn.ReLU(inplace=True)),
            ("10", nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)),
            ("relu4", nn.ReLU(inplace=True)),
            ("12", nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)),
            ("relu5", nn.ReLU(inplace=True)),
            ("pool5", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
        ]))
        self.classifier = nn.Sequential(OrderedDict([
            ("0", nn.Linear(256 * 6 * 6, 4096)),
            ("relu6", nn.ReLU(inplace=True)),
            ("drop6", nn.Dropout()),
            ("3", nn.Linear(4096, 4096)),
            ("relu7", nn.ReLU(inplace=True)),
            ("drop7", nn.Dropout())]))

        self.classifier_l = nn.Linear(512, n_classes)
        self.p_logvar = nn.Sequential(nn.Linear(4096, 512),
                                      nn.ReLU())
        self.p_mu = nn.Sequential(nn.Linear(4096, 512),
                                  nn.LeakyReLU())

    def get_params(self, base_lr):
        return [{"params": self.features.parameters(), "lr": 0.},
                {"params": chain(self.classifier.parameters(), self.classifier_l.parameters(), self.p_logvar.parameters(), self.p_mu.parameters()
                                 ), "lr": base_lr}]

    def is_patch_based(self):
        return False

    def forward(self, x, train=False):
        end_points={}
        x = self.features(x *57.6)  #57.6 is the magic number needed to bring torch data back to the range of caffe data, based on used std
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        logvar = self.p_logvar(x)
        mu = self.p_mu(x)
        end_points['logvar'] = logvar
        end_points['mu'] = mu

        if train:
            x = reparametrize(mu, logvar)
        else:
            x = mu

        end_points['Embedding'] = x
        x = self.classifier_l(x)
        end_points['Predictions'] = nn.functional.softmax(input=x, dim=-1)


        return x

def caffenet_L2D(classes):
    model = AlexNetCaffe(classes)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, .1)
            nn.init.constant_(m.bias, 0.)

    # state_dict = torch.load(os.path.join(os.path.dirname(__file__), "pretrained/alexnet_caffe.pth.tar"))
    # del state_dict["classifier.fc8.weight"]
    # del state_dict["classifier.fc8.bias"]
    # model.load_state_dict(state_dict['state_dict'], strict=False)

    #state_dict = torch.load(os.path.join(os.path.dirname(__file__), "pretrained/alexnet_caffe.pth.tar"))['state_dict']
    state_dict = torch.load(os.path.join(os.path.dirname(__file__), "pretrained/alexnet_caffe.pth"))
    del state_dict["classifier.6.weight"]
    del state_dict["classifier.6.bias"]
    model.load_state_dict(state_dict, strict=False)
    return model

def caffenet(classes):
    # model2 = models.alexnet(pretrained=True)
    #
    # torch.save(model2.state_dict(), "pretrained/alexnet_caffe.pth.tar")

    model = AlexNet(classes)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, .1)
            nn.init.constant_(m.bias, 0.)

    state_dict = torch.load(os.path.join(os.path.dirname(__file__), "pretrained/finetune_py_trans.tar"), map_location='cuda')  # alexnet_caffe.pth.tar"))
    # for key in state_dict:
    #     print(key)
    state_dict["cls_head.weight"] = state_dict["classifier.6.weight"]
    state_dict["cls_head.bias"] = state_dict["classifier.6.bias"]
    del state_dict["classifier.6.weight"]
    del state_dict["classifier.6.bias"]
    # for key in state_dict:
    #     print(key)
    model.load_state_dict(state_dict, strict=False)

    return model