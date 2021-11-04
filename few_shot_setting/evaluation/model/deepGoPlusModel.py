# -*- coding: utf-8 -*-
"""
Created on May 13 9:32 2021

@author: Hanwen Xu

E-mail: xuhw20@mails.tsinghua.edu.cn

Pytorch version reimplementation of deep neural models used by deepgoplus paper.
"""
from torch import nn
import torch


class deepGoPlusModel(nn.Module):

    def __init__(self, input_nc=4, n_classes=1000, in_nc=512, max_kernels=129, hidden_dense=0, seqL=2000):
        """

        :param input_nc:
        :param in_nc:
        :param max_kernels:
        :param dense_num:
        :param seqL:
        """
        super(deepGoPlusModel, self).__init__()
        self.para_conv, self.para_pooling = [], []
        kernels = range(8, max_kernels, 8)
        self.kernel_num = len(kernels)
        for i in range(len(kernels)):
            exec("self.conv1d_{} = nn.Conv1d(in_channels=input_nc, out_channels=in_nc, kernel_size=kernels[i], padding=0, stride=1)".format(i))
            exec("self.pool1d_{} = nn.MaxPool1d(kernel_size=seqL - kernels[i] + 1, stride=1)".format(i))
        self.fc = []
        for i in range(hidden_dense):
            self.fc.append(nn.Linear(len(kernels)*in_nc, len(kernels)*in_nc))
        self.fc.append(nn.Linear(len(kernels)*in_nc, n_classes))
        self.fc = nn.Sequential(*self.fc)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x_list = []
        for i in range(self.kernel_num):
            exec("x_i = self.conv1d_{}(x)".format(i))
            exec("x_i = self.pool1d_{}(x_i)".format(i))
            exec("x_list.append(torch.squeeze(x_i))")
        x1 = torch.cat(tuple(x_list), dim=1)
        x2 = self.fc(x1)
        return self.sigmoid(x2)


