# python3
# -*- coding: utf-8 -*-
# @Time    : 2021/12/15 20:33
# @Author  : Liupeng Lin
# @Email   : linliupeng@whu.edu.cn
# Copyright (C) 2021 Liupeng Lin. All Rights Reserved.


import os
import struct
import matplotlib.pyplot as plt
import scipy.io as scio
import math
import glob
import cmath
import numpy as np
import gdal
from PIL import Image
# import warnings
# warnings.filterwarnings("ignore")


def ReadData(filename):
    data = gdal.Open(filename)
    if data==None:
        print(filename + "文件无法打开")
        return
    im_width = data.RasterXSize
    im_height = data.RasterYSize
    im_bands = data.RasterCount
    im_data = data.ReadAsArray(0, 0, im_width, im_height)
    return im_data


def sar_paulishow(sar):
    [hei, wid, cha] = sar.shape
    sarpauli = np.zeros((hei, wid, 3), dtype=np.float)
    n = math.sqrt(2)
    sarpauli[:, :, 2] = sar[:, :, 0] / (np.mean(sar[:, :, 0]) * n)
    sarpauli[:, :, 0] = sar[:, :, 4] / (np.mean(sar[:, :, 4]) * n)
    sarpauli[:, :, 1] = sar[:, :, 8] / (np.mean(sar[:, :, 8]) * n)
    plt.axis('off')
    plt.imshow(sarpauli)
    plt.show()
    return sarpauli


def sar_assign(sar_value):
    [hei, wid, cha] = sar_value.shape
    sar = np.zeros((hei, wid, cha), dtype=np.complex)
    sar[:, :, 0] = abs(sar_value[:, :, 0])
    sar[:, :, 1] = sar_value[:, :, 1] + 1j*sar_value[:, :, 2]
    sar[:, :, 2] = sar_value[:, :, 3] + 1j*sar_value[:, :, 4]
    sar[:, :, 3] = sar_value[:, :, 1] - 1j*sar_value[:, :, 2]
    sar[:, :, 4] = abs(sar_value[:, :, 5])
    sar[:, :, 5] = sar_value[:, :, 6] + 1j*sar_value[:, :, 7]
    sar[:, :, 6] = sar_value[:, :, 3] - 1j*sar_value[:, :, 4]
    sar[:, :, 7] = sar_value[:, :, 6] - 1j*sar_value[:, :, 7]
    sar[:, :, 8] = abs(sar_value[:, :, 8])
    return sar


def sar_extract(sar):
    [hei, wid, cha] = sar.shape
    sar_value = np.zeros((hei, wid, cha), dtype=np.float)
    sar_value[:, :, 0] = sar[:, :, 0]
    sar_value[:, :, 1] = sar[:, :, 1].real
    sar_value[:, :, 2] = sar[:, :, 1].imag
    sar_value[:, :, 3] = sar[:, :, 2].real
    sar_value[:, :, 4] = sar[:, :, 2].imag
    sar_value[:, :, 5] = sar[:, :, 4]
    sar_value[:, :, 6] = sar[:, :, 5].real
    sar_value[:, :, 7] = sar[:, :, 5].imag
    sar_value[:, :, 8] = sar[:, :, 8]
    return sar_value


def mat_reshape(mat_in, hei_out, wid_out):
    [num, cha, hei, wid] = mat_in.shape
    mat_out = np.zeros((hei_out, wid_out, cha), dtype=np.float)
    for i in range(int(wid_out/wid)):
        for j in range(int(hei_out/hei)):
            mat_out[j*hei:(j+1)*hei, i*wid:(i+1)*wid, :] = (mat_in[j*int(wid_out/wid)+i, :, :, :].reshape(cha, hei, wid)).transpose(1, 2, 0)
    return mat_out


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


def C3_readbin(filepath):
    # 设置参数
    txt_i = 0
    datatype = 'C'
    # 读取输入路径下的文件名
    idtxt = os.path.join(filepath, 'config.txt')
    idt11 = os.path.join(filepath, '{}{}'.format(datatype, '11.bin'))
    idt22 = os.path.join(filepath, '{}{}'.format(datatype, '22.bin'))
    idt33 = os.path.join(filepath, '{}{}'.format(datatype, '33.bin'))
    idt12r = os.path.join(filepath, '{}{}'.format(datatype, '12_real.bin'))
    idt12i = os.path.join(filepath, '{}{}'.format(datatype, '12_imag.bin'))
    idt13r = os.path.join(filepath, '{}{}'.format(datatype, '13_real.bin'))
    idt13i = os.path.join(filepath, '{}{}'.format(datatype, '13_imag.bin'))
    idt23r = os.path.join(filepath, '{}{}'.format(datatype, '23_real.bin'))
    idt23i = os.path.join(filepath, '{}{}'.format(datatype, '23_imag.bin'))

    # 打开输入路径下的文件
    file_idt11 = open(idt11, 'rb')
    file_idt22 = open(idt22, 'rb')
    file_idt33 = open(idt33, 'rb')
    file_idt12r = open(idt12r, 'rb')
    file_idt12i = open(idt12i, 'rb')
    file_idt13r = open(idt13r, 'rb')
    file_idt13i = open(idt13i, 'rb')
    file_idt23r = open(idt23r, 'rb')
    file_idt23i = open(idt23i, 'rb')

    fidin = open(idtxt)
    a = np.zeros(2, dtype=np.int)
    while True:
        tline = fidin.readline()
        if not tline:
            break
        elif is_number(tline[0]):
            a[txt_i] = int(tline)
            txt_i = txt_i+1
    Nrow = a[0]
    Ncol = a[1]

    # 读取二进制文件
    sar_value = np.zeros((Nrow, Ncol, 9), dtype=np.float32)
    for i in range(Nrow):
        for j in range(Ncol):
            sar_value[i, j, 0] = struct.unpack('f', file_idt11.read(4))[0]
            sar_value[i, j, 1] = struct.unpack('f', file_idt12r.read(4))[0]
            sar_value[i, j, 2] = struct.unpack('f', file_idt12i.read(4))[0]
            sar_value[i, j, 3] = struct.unpack('f', file_idt13r.read(4))[0]
            sar_value[i, j, 4] = struct.unpack('f', file_idt13i.read(4))[0]
            sar_value[i, j, 5] = struct.unpack('f', file_idt22.read(4))[0]
            sar_value[i, j, 6] = struct.unpack('f', file_idt23r.read(4))[0]
            sar_value[i, j, 7] = struct.unpack('f', file_idt23i.read(4))[0]
            sar_value[i, j, 8] = struct.unpack('f', file_idt33.read(4))[0]
    sar = sar_assign(sar_value)

    return sar


def T3_readbin(filepath):
    # 设置参数
    txt_i = 0
    datatype = 'T'
    # 读取输入路径下的文件名
    idtxt = os.path.join(filepath, 'config.txt')
    idt11 = os.path.join(filepath, '{}{}'.format(datatype, '11.bin'))
    idt22 = os.path.join(filepath, '{}{}'.format(datatype, '22.bin'))
    idt33 = os.path.join(filepath, '{}{}'.format(datatype, '33.bin'))
    idt12r = os.path.join(filepath, '{}{}'.format(datatype, '12_real.bin'))
    idt12i = os.path.join(filepath, '{}{}'.format(datatype, '12_imag.bin'))
    idt13r = os.path.join(filepath, '{}{}'.format(datatype, '13_real.bin'))
    idt13i = os.path.join(filepath, '{}{}'.format(datatype, '13_imag.bin'))
    idt23r = os.path.join(filepath, '{}{}'.format(datatype, '23_real.bin'))
    idt23i = os.path.join(filepath, '{}{}'.format(datatype, '23_imag.bin'))

    # 打开输入路径下的文件
    file_idt11 = open(idt11, 'rb')
    file_idt22 = open(idt22, 'rb')
    file_idt33 = open(idt33, 'rb')
    file_idt12r = open(idt12r, 'rb')
    file_idt12i = open(idt12i, 'rb')
    file_idt13r = open(idt13r, 'rb')
    file_idt13i = open(idt13i, 'rb')
    file_idt23r = open(idt23r, 'rb')
    file_idt23i = open(idt23i, 'rb')

    fidin = open(idtxt)
    a = np.zeros(2, dtype=np.int)
    while True:
        tline = fidin.readline()
        if not tline:
            break
        elif is_number(tline[0]):
            a[txt_i] = int(tline)
            txt_i = txt_i+1
    Nrow = a[0]
    Ncol = a[1]

    # 读取二进制文件
    sar_value = np.zeros((Nrow, Ncol, 9), dtype=np.float32)
    for i in range(Nrow):
        for j in range(Ncol):
            sar_value[i, j, 0] = struct.unpack('f', file_idt11.read(4))[0]
            sar_value[i, j, 1] = struct.unpack('f', file_idt12r.read(4))[0]
            sar_value[i, j, 2] = struct.unpack('f', file_idt12i.read(4))[0]
            sar_value[i, j, 3] = struct.unpack('f', file_idt13r.read(4))[0]
            sar_value[i, j, 4] = struct.unpack('f', file_idt13i.read(4))[0]
            sar_value[i, j, 5] = struct.unpack('f', file_idt22.read(4))[0]
            sar_value[i, j, 6] = struct.unpack('f', file_idt23r.read(4))[0]
            sar_value[i, j, 7] = struct.unpack('f', file_idt23i.read(4))[0]
            sar_value[i, j, 8] = struct.unpack('f', file_idt33.read(4))[0]
    sar = sar_assign(sar_value)

    return sar


def C3_writebin(sar, savepath):
    str = '---------'
    datatype = 'C'
    sar_value = sar_extract(sar)
    [Nrow, Ncol, Ncha] = sar_value.shape
    savepath_new = os.path.join(savepath, '{}{}'.format(datatype, '3'))
    if not os.path.exists(savepath_new):
        os.mkdir(savepath_new)
    with open(os.path.join(savepath_new, 'config.txt'), 'w') as id0:
        id0.write('Nrow\n%d\n%s\nNcol\n%d\n%s\n' % (Nrow, str, Ncol, str))
        id0.write('PolarCase\nmonostatic\n%s\nPolarType\nfull\n' % (str))

    id1 = open(os.path.join(savepath_new, '{}{}'.format(datatype, '11.bin')), 'wb+')
    id2 = open(os.path.join(savepath_new, '{}{}'.format(datatype, '12_real.bin')), 'wb+')
    id3 = open(os.path.join(savepath_new, '{}{}'.format(datatype, '12_imag.bin')), 'wb+')
    id4 = open(os.path.join(savepath_new, '{}{}'.format(datatype, '13_real.bin')), 'wb+')
    id5 = open(os.path.join(savepath_new, '{}{}'.format(datatype, '13_imag.bin')), 'wb+')
    id6 = open(os.path.join(savepath_new, '{}{}'.format(datatype, '22.bin')), 'wb+')
    id7 = open(os.path.join(savepath_new, '{}{}'.format(datatype, '23_real.bin')), 'wb+')
    id8 = open(os.path.join(savepath_new, '{}{}'.format(datatype, '23_imag.bin')), 'wb+')
    id9 = open(os.path.join(savepath_new, '{}{}'.format(datatype, '33.bin')), 'wb+')

    for i in range(Nrow):
        for j in range(Ncol):
            id1.write(struct.pack('f', sar_value[i, j, 0]))
            id2.write(struct.pack('f', sar_value[i, j, 1]))
            id3.write(struct.pack('f', sar_value[i, j, 2]))
            id4.write(struct.pack('f', sar_value[i, j, 3]))
            id5.write(struct.pack('f', sar_value[i, j, 4]))
            id6.write(struct.pack('f', sar_value[i, j, 5]))
            id7.write(struct.pack('f', sar_value[i, j, 6]))
            id8.write(struct.pack('f', sar_value[i, j, 7]))
            id9.write(struct.pack('f', sar_value[i, j, 8]))
    # return print('Completed.')


def T3_writebin(sar, savepath):
    str = '---------'
    datatype = 'T'
    sar_value = sar_extract(sar)
    [Nrow, Ncol, Ncha] = sar_value.shape
    savepath_new = os.path.join(savepath, '{}{}'.format(datatype, '3'))
    if not os.path.exists(savepath_new):
        os.mkdir(savepath_new)
    with open(os.path.join(savepath_new, 'config.txt'), 'w') as id0:
        id0.write('Nrow\n%d\n%s\nNcol\n%d\n%s\n' % (Nrow, str, Ncol, str))
        id0.write('PolarCase\nmonostatic\n%s\nPolarType\nfull\n' % (str))

    id1 = open(os.path.join(savepath_new, '{}{}'.format(datatype, '11.bin')), 'wb+')
    id2 = open(os.path.join(savepath_new, '{}{}'.format(datatype, '12_real.bin')), 'wb+')
    id3 = open(os.path.join(savepath_new, '{}{}'.format(datatype, '12_imag.bin')), 'wb+')
    id4 = open(os.path.join(savepath_new, '{}{}'.format(datatype, '13_real.bin')), 'wb+')
    id5 = open(os.path.join(savepath_new, '{}{}'.format(datatype, '13_imag.bin')), 'wb+')
    id6 = open(os.path.join(savepath_new, '{}{}'.format(datatype, '22.bin')), 'wb+')
    id7 = open(os.path.join(savepath_new, '{}{}'.format(datatype, '23_real.bin')), 'wb+')
    id8 = open(os.path.join(savepath_new, '{}{}'.format(datatype, '23_imag.bin')), 'wb+')
    id9 = open(os.path.join(savepath_new, '{}{}'.format(datatype, '33.bin')), 'wb+')

    for i in range(Nrow):
        for j in range(Ncol):
            id1.write(struct.pack('f', sar_value[i, j, 0]))
            id2.write(struct.pack('f', sar_value[i, j, 1]))
            id3.write(struct.pack('f', sar_value[i, j, 2]))
            id4.write(struct.pack('f', sar_value[i, j, 3]))
            id5.write(struct.pack('f', sar_value[i, j, 4]))
            id6.write(struct.pack('f', sar_value[i, j, 5]))
            id7.write(struct.pack('f', sar_value[i, j, 6]))
            id8.write(struct.pack('f', sar_value[i, j, 7]))
            id9.write(struct.pack('f', sar_value[i, j, 8]))
    # return print('Completed.')


def c2t(c):
    '''
    :param c: Covariance matrix of PolSAR.
    :return t: Coherent matrix of PolSAR.
    '''

    sar_value = sar_extract(c)
    [row, col, cha] = sar_value.shape
    t_val = np.zeros((row, col, cha), dtype=np.float32)
    t_val[:, :, 0] = (c[:, :, 0] + c[:, :, 8] + c[:, :, 3] * 2) / 2
    t_val[:, :, 1] = (c[:, :, 0] - c[:, :, 8]) / 2
    t_val[:, :, 2] = - c[:, :, 4]
    t_val[:, :, 3] = (c[:, :, 1] + c[:, :, 6]) / math.sqrt(2)
    t_val[:, :, 4] = (c[:, :, 2] - c[:, :, 7]) / math.sqrt(2)
    t_val[:, :, 5] = (c[:, :, 0] + c[:, :, 8] - c[:, :, 3] * 2) / 2
    t_val[:, :, 6] = (c[:, :, 1] - c[:, :, 6]) / math.sqrt(2)
    t_val[:, :, 7] = (c[:, :, 2] + c[:, :, 7]) / math.sqrt(2)
    t_val[:, :, 8] = c[:, :, 5]
    t = sar_assign(t_val)

    return t


def C2_writebin(sar, savepath):
    str = '---------'
    datatype = 'C'
    sar_value = C2_extract(sar)
    [Nrow, Ncol, Ncha] = sar_value.shape
    savepath_new = os.path.join(savepath, '{}{}'.format(datatype, '2'))
    if not os.path.exists(savepath_new):
        os.mkdir(savepath_new)
    with open(os.path.join(savepath_new, 'config.txt'), 'w') as id0:
        id0.write('Nrow\n%d\n%s\nNcol\n%d\n%s\n' % (Nrow, str, Ncol, str))
        id0.write('PolarCase\nmonostatic\n%s\nPolarType\npp2\n' % (str))

    id1 = open(os.path.join(savepath_new, '{}{}'.format(datatype, '11.bin')), 'wb+')
    id2 = open(os.path.join(savepath_new, '{}{}'.format(datatype, '12_real.bin')), 'wb+')
    id3 = open(os.path.join(savepath_new, '{}{}'.format(datatype, '12_imag.bin')), 'wb+')
    id4 = open(os.path.join(savepath_new, '{}{}'.format(datatype, '22.bin')), 'wb+')

    for i in range(Nrow):
        for j in range(Ncol):
            id1.write(struct.pack('f', sar_value[i, j, 0]))
            id2.write(struct.pack('f', sar_value[i, j, 1]))
            id3.write(struct.pack('f', sar_value[i, j, 2]))
            id4.write(struct.pack('f', sar_value[i, j, 3]))
    # return print('Completed.')


def C2_assign(sar_value):
    [hei, wid, cha] = sar_value.shape
    sar = np.zeros((hei, wid, cha), dtype=np.complex)
    sar[:, :, 0] = abs(sar_value[:, :, 0])
    sar[:, :, 1] = sar_value[:, :, 1] + 1j*sar_value[:, :, 2]
    sar[:, :, 2] = sar_value[:, :, 1] - 1j*sar_value[:, :, 2]
    sar[:, :, 3] = abs(sar_value[:, :, 3])
    return sar


def C2_readbin(filepath):
    # 设置参数
    txt_i = 0
    datatype = 'C'
    # 读取输入路径下的文件名
    idtxt = os.path.join(filepath, 'config.txt')
    idt11 = os.path.join(filepath, '{}{}'.format(datatype, '11.bin'))
    idt22 = os.path.join(filepath, '{}{}'.format(datatype, '22.bin'))
    idt12r = os.path.join(filepath, '{}{}'.format(datatype, '12_real.bin'))
    idt12i = os.path.join(filepath, '{}{}'.format(datatype, '12_imag.bin'))

    # 打开输入路径下的文件
    file_idt11 = open(idt11, 'rb')
    file_idt22 = open(idt22, 'rb')
    file_idt12r = open(idt12r, 'rb')
    file_idt12i = open(idt12i, 'rb')

    fidin = open(idtxt)
    a = np.zeros(2, dtype=np.int)
    while True:
        tline = fidin.readline()
        if not tline:
            break
        elif is_number(tline[0]):
            a[txt_i] = int(tline)
            txt_i = txt_i+1
    Nrow = a[0]
    Ncol = a[1]

    # 读取二进制文件
    sar_value = np.zeros((Nrow, Ncol, 4), dtype=np.float32)
    for i in range(Nrow):
        for j in range(Ncol):
            sar_value[i, j, 0] = struct.unpack('f', file_idt11.read(4))[0]
            sar_value[i, j, 1] = struct.unpack('f', file_idt12r.read(4))[0]
            sar_value[i, j, 2] = struct.unpack('f', file_idt12i.read(4))[0]
            sar_value[i, j, 3] = struct.unpack('f', file_idt22.read(4))[0]
    sar = C2_assign(sar_value)
    return sar





def C2_extract(sar):
    [hei, wid, cha] = sar.shape
    sar_value = np.zeros((hei, wid, cha), dtype=np.float)
    sar_value[:, :, 0] = sar[:, :, 0]
    sar_value[:, :, 1] = sar[:, :, 1].real
    sar_value[:, :, 2] = sar[:, :, 1].imag
    sar_value[:, :, 3] = sar[:, :, 3]
    return sar_value


def C2_img2bin(C2_path, save_path):
    C11 = ReadData(glob.glob(os.path.join(C2_path, '*C11*.img'))[0])
    C12_real = ReadData(glob.glob(os.path.join(C2_path, '*C12_real*.img'))[0])
    C12_imag = ReadData(glob.glob(os.path.join(C2_path, '*C12_imag*.img'))[0])
    C22 = ReadData(glob.glob(os.path.join(C2_path, '*C22*.img'))[0])
    C2 = C2_assign(np.stack((C11, C12_real, C12_imag, C22,), axis=2))
    C2_writebin(C2, save_path)

