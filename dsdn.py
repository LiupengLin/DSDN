# python3
# -*- coding: utf-8 -*-
# @Time    : 2023/9/17 20:31
# @Author  : Liupeng Lin
# @Email   : linliupeng@whu.edu.cn
# Copyright (C) 2023 Liupeng Lin. All Rights Reserved.


import torch
import torch.nn as nn
import torch.nn.functional as F


class make_dense(nn.Module):
    def __init__(self, nFeat, growthRate):
        super(make_dense, self).__init__()
        self.conv_dense = nn.Sequential(
            nn.Conv2d(nFeat, growthRate, kernel_size=3,  stride=1, padding=1, bias=True), nn.PReLU())

    def forward(self, x):
        out1 = self.conv_dense(x)
        out = torch.cat((x, out1), 1)
        return out


class RDB(nn.Module):
    def __init__(self, nFeat, nDense, growthRate):
        super(RDB, self).__init__()
        nFeat_ = nFeat
        modules = []
        for i in range(nDense):
            modules.append(make_dense(nFeat_, growthRate))
            nFeat_ += growthRate
            self.dense_layers = nn.Sequential(*modules)
            self.conv_1x1 = nn.Conv2d(nFeat_, nFeat, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        out1 = self.conv_1x1(self.dense_layers(x))
        out = torch.add(x, out1)
        return out


class CAL(nn.Module):
    def __init__(self, nFeat, ratio=16):
        super(CAL, self).__init__()
        self.cal_conv1 = nn.Sequential(
            nn.Conv2d(nFeat, nFeat, kernel_size=3,  stride=1, padding=1, bias=True), nn.PReLU())
        self.cal_conv2 = nn.Sequential(
            nn.Conv2d(nFeat, nFeat, kernel_size=1, padding=0, bias=True), nn.PReLU())
        self.cal_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cal_fc1 = nn.Sequential(
            nn.Conv2d(nFeat, nFeat // ratio, 1, padding=0, bias=True), nn.PReLU())
        self.cal_fc2 = nn.Sequential(
            nn.Conv2d(nFeat // ratio, nFeat, 1, padding=0, bias=True), nn.Sigmoid())

    def forward(self, x):
        cal_weight_avg = self.cal_fc2(self.cal_fc1(self.cal_avg_pool(x)))
        out = self.cal_conv2(torch.mul(x, cal_weight_avg))
        return out


class SAL(nn.Module):
    def __init__(self, nFeat):
        super(SAL, self).__init__()
        self.sal_conv1 = nn.Sequential(
            nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, groups=nFeat, bias=True), nn.PReLU())
        self.sal_conv1x1 = nn.Sequential(
            nn.Conv2d(nFeat, 1, kernel_size=1, padding=0, bias=True), nn.Sigmoid())
        self.sal_conv2 = nn.Sequential(
            nn.Conv2d(nFeat, nFeat, kernel_size=3, stride=1, padding=1, bias=True), nn.PReLU())

    def forward(self, x):
        sal_weight = self.sal_conv1x1(self.sal_conv1(x))
        out = self.sal_conv2(x * sal_weight)
        return out


class RCSA(nn.Module):
    def __init__(self, nFeat):
        super(RCSA, self).__init__()
        self.rcsa_cal = CAL(nFeat)
        self.rcsa_sal = SAL(nFeat)
        self.rcsa_conv1 = nn.Sequential(
            nn.Conv2d(nFeat, nFeat, kernel_size=3, stride=1, padding=1, bias=True), nn.PReLU())

    def forward(self, x):
        out1 = self.rcsa_cal(x)
        out2 = self.rcsa_sal(x)
        out = torch.add(x, self.rcsa_conv1(torch.add(out1, out2)))
        return out


class CroA(nn.Module):
    """ Cross Attention
    """
    def __init__(self, nFeat):
        super(CroA, self).__init__()
        self.croa_conv1 = nn.Sequential(
            nn.Conv2d(nFeat, nFeat, kernel_size=3,  stride=1, padding=1, bias=True), nn.PReLU())
        self.croa_conv2 = nn.Sequential(
            nn.Conv2d(nFeat, nFeat, kernel_size=3,  stride=1, padding=1, bias=True), nn.PReLU())
        self.croa_conv3 = nn.Sequential(
            nn.Conv2d(nFeat, nFeat, kernel_size=3,  stride=1, padding=1, bias=True), nn.Sigmoid())
        self.croa_conv4 = nn.Sequential(
            nn.Conv2d(nFeat, nFeat, kernel_size=3,  stride=1, padding=1, bias=True), nn.PReLU())
        self.croa_conv1x1_1 = nn.Sequential(
            nn.Conv2d(nFeat, 1, kernel_size=1, padding=0, bias=True), nn.Sigmoid())

        self.croa_conv1x1_2 = nn.Sequential(
            nn.Conv2d(nFeat*2, nFeat, kernel_size=1, padding=0, bias=True), nn.PReLU())

    def forward(self, lr, hr):
        lr_fea = self.croa_conv1(lr)
        hr_fea = self.croa_conv2(hr)
        out = self.croa_conv4(self.croa_conv1x1_2(torch.cat((torch.mul(lr_fea, self.croa_conv1x1_1(hr_fea)), torch.mul(hr_fea, self.croa_conv3(lr_fea))), 1)))
        return out


class DSDN_i(nn.Module):
    def __init__(self, args):
        super(DSDN_i, self).__init__()
        ncha_ni1 = args.ncha_ni1
        nDense = args.nDense
        nFeat = args.nFeat
        growthRate = args.growthRate

        self.conv1 = nn.Conv2d(ncha_ni1, nFeat, kernel_size=3, padding=1, bias=True)

        self.conv2 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)

        self.RDB1 = RDB(nFeat, nDense, growthRate)
        self.RDB2 = RDB(nFeat, nDense, growthRate)
        self.RDB3 = RDB(nFeat, nDense, growthRate)

        self.rcsa1 = RCSA(nFeat)
        self.rcsa2 = RCSA(nFeat)
        self.rcsa3 = RCSA(nFeat)
        self.croa1 = CroA(nFeat)

        self.GFF_1x1 = nn.Conv2d(nFeat*3, nFeat, kernel_size=1, padding=0, bias=True)
        self.GFF_3x3 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)

        self.conv3 = nn.Conv2d(nFeat, ncha_ni1, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.rcsa1(self.RDB1(out2))
        out4 = self.rcsa2(self.RDB2(out3))
        out5 = self.rcsa3(self.RDB3(out4))

        out6 = self.GFF_1x1(torch.cat((out3, out4, out5), 1))
        out7 = self.GFF_3x3(out6)
        out8 = torch.add(out7, out1)
        out = self.conv3(self.croa1(out8, out1))
        return out


class DSDN_p(nn.Module):
    def __init__(self, args):
        super(DSDN_p, self).__init__()
        ncha_np1 = args.ncha_np1
        nDense = args.nDense
        nFeat = args.nFeat
        growthRate = args.growthRate

        self.conv1 = nn.Conv2d(ncha_np1, nFeat, kernel_size=3, padding=1, bias=True)

        self.conv2 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)

        self.RDB1 = RDB(nFeat, nDense, growthRate)
        self.RDB2 = RDB(nFeat, nDense, growthRate)
        self.RDB3 = RDB(nFeat, nDense, growthRate)

        self.rcsa1 = RCSA(nFeat)
        self.rcsa2 = RCSA(nFeat)
        self.rcsa3 = RCSA(nFeat)
        self.croa1 = CroA(nFeat)

        self.GFF_1x1 = nn.Conv2d(nFeat*3, nFeat, kernel_size=1, padding=0, bias=True)
        self.GFF_3x3 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)

        self.conv3 = nn.Conv2d(nFeat, ncha_np1, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.rcsa1(self.RDB1(out2))
        out4 = self.rcsa2(self.RDB2(out3))
        out5 = self.rcsa3(self.RDB3(out4))

        out6 = self.GFF_1x1(torch.cat((out3, out4, out5), 1))
        out7 = self.GFF_3x3(out6)
        out8 = torch.add(out7, out1)
        out = self.conv3(self.croa1(out8, out1))
        return out


class DSDN(nn.Module):
    def __init__(self, args):
        super(DSDN, self).__init__()
        self.DSDN_i = DSDN_i(args)
        self.DSDN_p = DSDN_p(args)

    def forward(self, i, p):
        out1 = self.DSDN_i(i)
        out2 = self.DSDN_p(p)
        return out1, out2

