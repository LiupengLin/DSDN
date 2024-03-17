# python3
# -*- coding: utf-8 -*-
# @Time    : 2023/9/19 16:33
# @Author  : Liupeng Lin
# @Email   : linliupeng@whu.edu.cn
# Copyright (C) 2023 Liupeng Lin. All Rights Reserved.

import re
import time
import glob
import torch
import argparse
import warnings
from dltoolbox import *
from sartoolbox import *
from dsdn import DSDN
import h5py
warnings.filterwarnings("ignore")


# parsers
parser = argparse.ArgumentParser(description='DSDN')
parser.add_argument('--model', default='dsdn', type=str, help='choose path of model')
parser.add_argument('--epoch', default=5, type=int, help='number of train epoch')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for Adam')
parser.add_argument('--gpu', default='0', type=str, help='gpu id')
parser.add_argument('--nDense', type=int, default=6, help='nDenselayer of RDB')
parser.add_argument('--growthRate', type=int, default=32, help='growthRate of dense net')
parser.add_argument('--nFeat', type=int, default=64,  help='number of feature maps')
parser.add_argument('--ncha_ni1', type=int, default=4, help='number of hr channels to use')
parser.add_argument('--ncha_np1', type=int, default=3, help='number of lr channels to use')
parser.add_argument('--ncha_ni2', type=int, default=4, help='number of hr channels to use')
parser.add_argument('--ncha_np2', type=int, default=3, help='number of lr channels to use')
parser.add_argument('--ncha_cd', type=int, default=1, help='number of lr channels to use')
parser.add_argument('--border_size', default=10, type=int, help='border size')
parser.add_argument('--cm_dir', default='data/test/cm/', type=str, help='path of cm data')
parser.add_argument('--pd_dir', default='data/test/pd/', type=str, help='path of pd data')
parser.add_argument('--result_path', default='data/eval/', type=str, help='path of result')
args = parser.parse_args()


cuda = torch.cuda.is_available()
nGPU = torch.cuda.device_count()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

result_dir = os.path.join(args.result_path, args.model)

if not os.path.exists(result_dir):
    os.mkdir(result_dir)

savedir = os.path.join('model', args.model)


def find_checkpoint(savedir):
    file_list = glob.glob(os.path.join(savedir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for m in file_list:
            result = re.findall('.*model_(.*).pth.*', m)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


def ReadHDF(path):
    mat = h5py.File(path, 'r+')
    pymat = np.transpose(mat[list(mat.keys())[0]])
    return pymat


def main():
    # load model
    model = DSDN(args)
    initial_epoch = find_checkpoint(savedir=savedir)
    if initial_epoch > 0:
        print('==> Resuming by loading epoch %03d' % initial_epoch)
        model = torch.load(os.path.join(savedir, 'model_%03d.pth' % initial_epoch))
    model.eval()

    if cuda:
        model = model.cuda()

    # load test data
    data_list = glob.glob(os.path.join(args.cm_dir, '*'))
    for i in range(len(data_list)):
        savepath = str(data_list[i]).replace('test/cm', 'eval/'+args.model)

        if not os.path.exists(savepath):
            os.mkdir(savepath)

        CM = ReadHDF(os.path.join(str(data_list[i]), 'cm.mat'))
        PD = ReadHDF(os.path.join(str(data_list[i]), 'cm.mat').replace('cm', 'pd'))

        [hei_ori, wid_ori, cha_ori] = CM.shape

        patch_size_hei, patch_size_wid = ImgDivideSize(CM)
        CM_patch = ImgDividePad(CM, args.border_size, patch_size_hei, patch_size_wid)
        PD_patch = ImgDividePad(PD, args.border_size, patch_size_hei, patch_size_wid)

        # initial result
        [num_cm, cha_cm, hei_cm, wid_cm] = CM_patch.shape
        [num_pd, cha_pd, hei_pd, wid_pd] = PD_patch.shape
        CM_result = np.zeros((num_cm, cha_cm, hei_cm, wid_cm), dtype=np.float32)

        start_time = time.time()
        for n in range(num_cm):
            sub_CM = CM_patch[n, :, :, :]
            sub_PD = PD_patch[n, :, :, :]

            sub_CM = torch.from_numpy(sub_CM).contiguous().view(1, cha_cm, hei_cm, wid_cm)
            sub_PD = torch.from_numpy(sub_PD).contiguous().view(1, cha_pd, hei_pd, wid_pd)

            sub_CM = sub_CM.cuda()
            sub_PD = sub_PD.cuda()

            sub_out1,  sub_out2 = model(sub_CM, sub_PD)

            sub_out1 = sub_out1.cpu()

            sub_out1 = sub_out1.detach().numpy().astype(np.float32)
            CM_result[n, :, :, :] = sub_out1

        elapsed_time = time.time() - start_time
        print('processing time is %4.4f s' % elapsed_time)

        result_meg = ImgMergePad(CM_result, args.border_size, hei_ori, wid_ori)
        sar = C2_assign(result_meg)
        C2_writebin(sar, savepath)
        # sar_paulishow(sr)
        print('the export of result is complete')


if __name__ == '__main__':
    main()
