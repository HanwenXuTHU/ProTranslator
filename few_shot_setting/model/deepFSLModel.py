# -*- coding: utf-8 -*-
"""
Created on May 13 9:32 2021

@author: Hanwen Xu

E-mail: xuhw20@mails.tsinghua.edu.cn

Pytorch version reimplementation of deep neural models used by deepgoplus paper.
"""
from torch import nn
import torch


class deepFSLModel(nn.Module):

    def __init__(self,
                 input_nc=4,
                 n_classes=5101,
                 in_nc=512,
                 max_kernels=129,
                 seqL=2000,
                 hidden_dim=[6000],
                 emb_dim=768,
                 dropout=0.2):
        """

        :param input_nc:
        :param in_nc:
        :param max_kernels:
        :param dense_num:
        :param seqL:
        """
        super(deepFSLModel, self).__init__()
        self.para_conv, self.para_pooling = [], []
        kernels = range(8, max_kernels, 8)
        self.kernel_num = len(kernels)
        for i in range(len(kernels)):
            exec("self.conv1d_{} = nn.Conv1d(in_channels=input_nc, out_channels=in_nc, kernel_size=kernels[i], padding=0, stride=1)".format(i))
            exec("self.pool1d_{} = nn.MaxPool1d(kernel_size=seqL - kernels[i] + 1, stride=1)".format(i))
        self.fc = []
        self.hidden_dim = []
        self.hidden_dim.append(len(kernels)*in_nc)
        for i in range(len(hidden_dim)):
            self.hidden_dim.append(hidden_dim[i])
        self.hidden_dim.append(emb_dim)
        for i in range(len(self.hidden_dim) - 1):
            self.fc.append(nn.Linear(self.hidden_dim[i], self.hidden_dim[i + 1]))
            if i != len(self.hidden_dim) - 2:
                self.fc.append(nn.ReLU(inplace=True))
        self.fc_emb = nn.Sequential(*self.fc)
        self.fc_no_emb = nn.Sequential(*[nn.Linear(len(kernels)*in_nc, n_classes),
                          ])
        self.activation = torch.nn.Sigmoid()

    def forward(self, x, emb_tensor):
        x_list = []
        for i in range(self.kernel_num):
            exec("x_i = self.conv1d_{}(x)".format(i))
            exec("x_i = self.pool1d_{}(x_i)".format(i))
            exec("x_list.append(torch.squeeze(x_i))")
        x_cat = torch.cat(tuple(x_list), dim=1)
        x1 = self.fc_emb(x_cat)
        emb2 = emb_tensor.permute(1, 0)
        x2 = torch.mm(x1, emb2)
        x3 = self.fc_no_emb(x_cat)
        return self.activation(x2 + x3)


class onlyTextModel(nn.Module):

    def __init__(self,
                 input_nc=4,
                 n_classes=5101,
                 in_nc=512,
                 max_kernels=129,
                 seqL=2000,
                 hidden_dim=[6000],
                 emb_dim=768,
                 dropout=0.2):
        """

        :param input_nc:
        :param in_nc:
        :param max_kernels:
        :param dense_num:
        :param seqL:
        """
        super(onlyTextModel, self).__init__()
        self.fc = []
        self.hidden_dim = []
        self.hidden_dim.append(seqL*input_nc)
        for i in range(len(hidden_dim)):
            self.hidden_dim.append(hidden_dim[i])
        self.hidden_dim.append(emb_dim)
        for i in range(len(self.hidden_dim) - 1):
            self.fc.append(nn.Linear(self.hidden_dim[i], self.hidden_dim[i + 1]))
            if i != len(self.hidden_dim) - 2:
                self.fc.append(nn.ReLU(inplace=True))
        self.fc_emb = nn.Sequential(*self.fc)
        self.activation = torch.nn.Sigmoid()

    def forward(self, x, emb_tensor):
        x1 = self.fc_emb(x.view(x.size(0), -1))
        emb2 = emb_tensor.permute(1, 0)
        x2 = torch.mm(x1, emb2)
        return self.activation(x2)