from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import gc
import visdom
import os
import time
from os import listdir
from PIL import Image
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torchvision import utils, transforms, models
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader



# Some utilities
class VisdomLine():
    def __init__(self, vis, opts):
        self.vis = vis
        self.opts = opts
        self.win = None

    def Update(self, x, y):
        if self.win is None:
            self.win = self.vis.line(X=x, Y=y, opts=self.opts)
        else:
            self.vis.line(X=x, Y=y, opts=self.opts, win=self.win)

class VisdomImage():
    def __init__(self, vis, opts):
        self.vis = vis
        self.opts = opts
        self.win = None

    def Update(self, image):
        if self.win is None:
            self.win = self.vis.image(image, opts=self.opts)
        else:
            self.vis.image(image, opts=self.opts, win=self.win)

def LearningRateScheduler(optimizer, epoch, lr_decay=0.5, lr_decay_step=10):
    if epoch % lr_decay_step:
        return optimizer

    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay

    print('Learning rate is decreased by %f' % (lr_decay))

    return optimizer



# For data loading
class DataManager(Dataset):
    def __init__(self, path_content, path_style, random_crop=True):
        self.path_content = path_content
        self.path_style = path_style

        # Preprocessing for imagenet pre-trained network
        if random_crop:
            self.transform = transforms.Compose(
                [
                    transforms.RandomCrop((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ]
            )

        # Convert pre-processed images to original images
        self.restore = transforms.Compose(
            [
                transforms.Normalize(mean=[-2.118, -2.036, -1.804],
                                     std=[4.367, 4.464, 4.444]),
            ]
        )

        self.list_content = listdir(self.path_content)
        self.list_style = listdir(self.path_style)

        self.num_content = len(self.list_content)
        self.num_style = len(self.list_style)

        assert self.num_content > 0
        assert self.num_style > 0

        self.num = min(self.num_content, self.num_style)

        print('Content root : %s' % (self.path_content))
        print('Style root : %s' % (self.path_style))
        print('Number of content images : %d' % (self.num_content))
        print('Number of style images : %d' % (self.num_style))
        print('Dataset size : %d' % (self.num))

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        path_to_con = self.path_content + '/' + self.list_content[idx]
        path_to_sty = self.path_style + '/' + self.list_style[idx]

        img_con = Image.open(path_to_con)
        img_con = self.transform(img_con)

        img_sty = Image.open(path_to_sty)
        img_sty = self.transform(img_sty)

        sample = {'content': img_con, 'style': img_sty}

        return sample



"""
    Task 1. Define your neural network here.
"""
class StyleTransferNet(nn.Module):
    def __init__(self, w_style=0.01):
        super(StyleTransferNet, self).__init__()
        self.w_style = w_style
        self.loss_fn = nn.MSELoss()

        #############################
        # Encoder
        #############################
        pretrained_vgg19 = models.vgg19(pretrained=True)

        # fix weight of pretrained model
        for param in pretrained_vgg19.parameters():
            param.requires_grad = False

        # separate first layer for feature extraction
        vgg19_layers = list(pretrained_vgg19.features.children())
        self.enc_1 = nn.Sequential(*vgg19_layers[0:2])   #relu_1.1
        self.enc_2 = nn.Sequential(*vgg19_layers[2:7])   #relu_2.1
        self.enc_3 = nn.Sequential(*vgg19_layers[7:12])  #relu_3.1
        self.enc_4 = nn.Sequential(*vgg19_layers[12:21]) #relu_4.1
        self.enc = nn.Sequential(self.enc_1, self.enc_2, self.enc_3, self.enc_4)

        #############################
        # Decoder
        #############################
       	self.dec = nn.Sequential(
	    nn.ReflectionPad2d((1, 1, 1, 1)),
	    nn.Conv2d(512, 256, (3, 3)),
	    nn.ReLU(),
	    nn.Upsample(scale_factor=2, mode='nearest'),
	    nn.ReflectionPad2d((1, 1, 1, 1)),
	    nn.Conv2d(256, 256, (3, 3)),
	    nn.ReLU(),
	    nn.ReflectionPad2d((1, 1, 1, 1)),
	    nn.Conv2d(256, 256, (3, 3)),
	    nn.ReLU(),
	    nn.ReflectionPad2d((1, 1, 1, 1)),
	    nn.Conv2d(256, 256, (3, 3)),
	    nn.ReLU(),
	    nn.ReflectionPad2d((1, 1, 1, 1)),
	    nn.Conv2d(256, 128, (3, 3)),
	    nn.ReLU(),
	    nn.Upsample(scale_factor=2, mode='nearest'),
	    nn.ReflectionPad2d((1, 1, 1, 1)),
	    nn.Conv2d(128, 128, (3, 3)),
	    nn.ReLU(),
	    nn.ReflectionPad2d((1, 1, 1, 1)),
	    nn.Conv2d(128, 64, (3, 3)),
	    nn.ReLU(),
	    nn.Upsample(scale_factor=2, mode='nearest'),
	    nn.ReflectionPad2d((1, 1, 1, 1)),
	    nn.Conv2d(64, 64, (3, 3)),
	    nn.ReLU(),
	    nn.ReflectionPad2d((1, 1, 1, 1)),
	    nn.Conv2d(64, 3, (3, 3)),
	)
    
    def compute_mean_std(self, x):
        B, C, H, W = x.shape
        eps = 1e-5
        mean = x.view(B, C, -1).mean(dim=2).view(B, C, 1, 1)
        std  = x.view(B, C, -1).std(dim=2).view(B, C, 1, 1) + eps
        return mean, std

    def AdaINLayer(self, x, y):
        Bx, Cx, Hx, Wx = x.shape
        By, Cy, Hy, Wy = y.shape

        assert Bx == By
        assert Cx == Cy

        """
            Define your AdaIN layer in here
            
            output : the result of AdaIN operation
        """
        mean_x, std_x = self.compute_mean_std(x)
        mean_y, std_y = self.compute_mean_std(y)
        output = std_y*(x - mean_x)/std_x + mean_y
    
        return output

    def get_enc_features(self, img):
        feats = [img]
        for i in range(1, 5):
            layer = getattr(self, 'enc_{:d}'.format(i))
            feats.append(layer(feats[-1]))
        return feats

    def compute_style_loss(self, img_rcv, img_sty):
        feat_rcv = self.get_enc_features(img_rcv)
        feat_sty = self.get_enc_features(img_sty)
        loss = 0
        for i in range(1, 5):
            mean_rcv, std_rcv = self.compute_mean_std(feat_rcv[i])
            mean_sty, std_sty = self.compute_mean_std(feat_sty[i])
            loss = loss + self.loss_fn(mean_rcv, mean_sty) + self.loss_fn(std_rcv, std_sty)
        return loss
        

    def forward(self, x, y, alpha=1):
        B, C, H, W = x.shape

        """
            Define forward process using torch operations
            
            loss : content loss + style loss
            img_result : style transferred image
                         (output of the decoder network)
            alpha : trade-off between content and style at test time
        """
        enc_con = self.enc(x)
        enc_sty = self.enc(y)

        AdaIN = self.AdaINLayer(enc_con, enc_sty)
        t = alpha*AdaIN + (1-alpha)*enc_con
        img_result = self.dec(t)

        # content_loss + w_style*style_loss
        loss = self.loss_fn(self.enc(img_result), t) + self.w_style*self.compute_style_loss(img_result, y)

        return loss, img_result



