# python3
# -*- coding: utf-8 -*-
# @Time    : 2023/9/17 20:52
# @Author  : Liupeng Lin
# @Email   : linliupeng@whu.edu.cn
# Copyright (C) 2023 Liupeng Lin. All Rights Reserved.

import os
import time
import h5py
import glob
import re
import warnings
import argparse
import numpy as np
import torch
from dsdn import DSDN
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from dataset import *
from torch.utils.data import DataLoader
from torch.autograd import Variable
import gc
from loss import *
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='DSDN')
parser.add_argument('--model', default='dsdn', type=str, help='choose path of model')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--patch_size', default=40, type=int, help='patch size')
parser.add_argument('--epoch', default=100, type=int, help='number of train epoch')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for Adam')
parser.add_argument('--train_data', default='data/h5/train_dsdn0921.h5', type=str, help='path of train data')
parser.add_argument('--gpu', default='0', type=str, help='gpu id')
parser.add_argument('--nDense', type=int, default=6, help='nDenselayer of RDB')
parser.add_argument('--growthRate', type=int, default=32, help='growthRate of dense net')
parser.add_argument('--nFeat', type=int, default=64,  help='number of feature maps')
parser.add_argument('--ncha_ni1', type=int, default=4, help='number of hr channels to use')
parser.add_argument('--ncha_np1', type=int, default=3, help='number of lr channels to use')
parser.add_argument('--ncha_ni2', type=int, default=4, help='number of hr channels to use')
parser.add_argument('--ncha_np2', type=int, default=3, help='number of lr channels to use')
parser.add_argument('--ncha_cd', type=int, default=1, help='number of lr channels to use')
args = parser.parse_args()

cuda = torch.cuda.is_available()
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

batch_size = args.batch_size
nepoch = args.epoch
savedir = os.path.join('model', args.model)

if not os.path.exists(savedir):
    os.mkdir(savedir)

log_header = [
    'epoch',
    'iteration',
    'train/loss',
]
if not os.path.exists(os.path.join(savedir, 'log.csv')):
    with open(os.path.join(savedir, 'log.csv'), 'w') as f:
        f.write(','.join(log_header) + '\n')


def find_checkpoint(savedir):
    file_list = glob.glob(os.path.join(savedir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for m in file_list:
            result = re.findall(".*model_(.*).pth.*", m)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


def main():
    # dataset generator
    print("==> Generating data")
    hf = h5py.File(args.train_data, 'r+')
    ni1 = np.float32(hf['data1'])
    np1 = np.float32(hf['data2'])
    np2 = np.float32(hf['data3'])
    cd = np.float32(hf['data4'])
    ni2 = np.float32(hf['label'])

    ni1 = torch.from_numpy(ni1).view(-1, args.ncha_ni1, args.patch_size, args.patch_size)
    np1 = torch.from_numpy(np1).view(-1, args.ncha_np1, args.patch_size, args.patch_size)
    np2 = torch.from_numpy(np2).view(-1, args.ncha_np2, args.patch_size, args.patch_size)
    cd = torch.from_numpy(cd).view(-1, args.ncha_cd, args.patch_size, args.patch_size)
    ni2 = torch.from_numpy(ni2).view(-1, args.ncha_ni2, args.patch_size, args.patch_size)

    train_set = DSDNDataset(ni1, np1, ni2, np2, cd)
    train_loader = DataLoader(dataset=train_set, num_workers=8, drop_last=True, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    print("==> Building model")
    model = DSDN(args)
    criterion1 = CharbonnierLoss()
    criterion2 = CharbonnierLoss()
    if cuda:
        print("==> Setting GPU")
        print("===> gpu id: '{}'".format(args.gpu))
        model = model.cuda()
        criterion1 = criterion1.cuda()
        criterion2 = criterion2.cuda()

    print("==> Setting optimizer")
    optimizer = optim.Adam(model.parameters(), lr=args.lr,  weight_decay=0)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.9)  # learning rates adjust

    initial_epoch = find_checkpoint(savedir=savedir)
    if initial_epoch > 0:
        print('==> Resuming by loading epoch %03d' % initial_epoch)
        model = torch.load(os.path.join(savedir, 'model_%03d.pth' % initial_epoch))

    if len(args.gpu) > 1:
        model = torch.nn.DataParallel(model)

    for epoch in range(initial_epoch, nepoch):
        scheduler.step(epoch)
        epoch_loss = 0
        start_time = time.time()

        # train
        model.train()
        num_train = len(train_loader.dataset)

        for iteration, batch in enumerate(train_loader):
            ni1_batch, np1_batch, ni2_batch, np2_batch, cd_batch = Variable(batch[0]), Variable(batch[1]), Variable(batch[2]), Variable(batch[3]), Variable(batch[4])
            if cuda:
                ni1_batch = ni1_batch.cuda()
                np1_batch = np1_batch.cuda()
                ni2_batch = ni2_batch.cuda()
                np2_batch = np2_batch.cuda()
                cd_batch = cd_batch.cuda()

            optimizer.zero_grad()

            out1, out2 = model(ni1_batch, np1_batch)
            loss1 = criterion1(torch.mul(out1, cd_batch), torch.mul(ni2_batch, cd_batch))
            loss2 = criterion2(torch.mul(out2, cd_batch), torch.mul(np2_batch, cd_batch))

            lambda1 = loss1.data / (loss1.data + loss2.data)
            lambda2 = loss2.data / (loss1.data + loss2.data)

            loss = lambda1 * loss1 + lambda2 * loss2

            print('%4d %4d / %4d loss = %2.6f' % (epoch + 1, iteration, train_set.noise_im1.size(0)/batch_size, loss.data))
            loss.backward()
            optimizer.step()
            with open(os.path.join(savedir, 'log.csv'), 'a') as file:
                log = [epoch, iteration] + [loss.data.item()]
                log = map(str, log)
                file.write(','.join(log) + '\n')
        if len(args.gpu) > 1:
            if ((epoch + 1) % 5) == 0:
                torch.save(model.module, os.path.join(savedir, 'model_%03d.pth' % (epoch + 1)))
        else:
            if ((epoch + 1) % 5) == 0:
                torch.save(model, os.path.join(savedir, 'model_%03d.pth' % (epoch + 1)))
        gc.collect()
        elapsed_time = time.time() - start_time
        print('epcoh = %4d , time is %4.4f s' % (epoch + 1, elapsed_time))


if __name__ == '__main__':
    main()