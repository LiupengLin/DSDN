# python3
# -*- coding: utf-8 -*-
# @Time    : 2021/11/9 21:39
# @Author  : Liupeng Lin
# @Email   : linliupeng@whu.edu.cn
# Copyright (C) 2021 Liupeng Lin. All Rights Reserved.


import numpy as np
import cv2
import gdal
import math


def mat2pymat(mat):
    [hei, wid, cha] = mat.shape
    mat_py = np.zeros((cha, hei, wid), dtype=np.float)
    for i in range(hei):
        for j in range(wid):
            mat_py[:, i, j] = mat[i, j, :]
    return mat_py


def pymat2mat(mat_py):
    [cha, hei, wid] = mat_py.shape
    mat = np.zeros((hei, wid, cha), dtype=np.float)
    for i in range(hei):
        for j in range(wid):
            mat[i, j, :] = mat_py[:, i, j]
    return mat


def mat_reshape(mat_in, hei_out, wid_out):
    [num, cha, hei, wid] = mat_in.shape
    mat_out = np.zeros((hei_out, wid_out, cha), dtype=np.float)
    for i in range(int(wid_out/wid)):
        for j in range(int(hei_out/hei)):
            mat_out[j*hei:(j+1)*hei, i*wid:(i+1)*wid, :] = (mat_in[j*int(wid_out/wid)+i, :, :, :].reshape(cha, hei, wid)).transpose(1, 2, 0)
            # pymat2mat(mat_in[j*int(wid_out/wid)+i, :, :, :].reshape(cha, hei, wid))
    return mat_out


def ImgDivideSize(img):
    # 返回一个1-512之间的整除数
    if len(img.shape) == 3:
        hei, wid, cha = img.shape
    elif len(img.shape) == 2:
        (hei, wid), cha = img.shape, 1
    else:
        print('the size of the input image is wrong!')

    patch_size_hei = hei
    patch_size_wid = wid

    for i in range(1, 513):
        if hei % i == 0:
            patch_size_hei = i

    for j in range(1, 513):
        if wid % j == 0:
            patch_size_wid = j

    return patch_size_hei, patch_size_wid


def ImgDivideSizeI2(img, scale):
    # 返回一个1-400之间的整除数
    if len(img.shape) == 3:
        hei, wid, cha = img.shape
    elif len(img.shape) == 2:
        (hei, wid), cha = img.shape, 1
    else:
        print('the size of the input image is wrong!')

    patch_size_hei = hei
    patch_size_wid = wid

    for i in range(1, 401):
        if (hei % i == 0) and (i % scale == 0):
            patch_size_hei = i

    for j in range(1, 401):
        if (wid % j == 0) and (j % scale == 0):
            patch_size_wid = j

    return patch_size_hei, patch_size_wid


def ImgDivideSizeI3(img, scale1, scale2):
    # 返回一个1-300之间的整除数
    if len(img.shape) == 3:
        hei, wid, cha = img.shape
    elif len(img.shape) == 2:
        (hei, wid), cha = img.shape, 1
    else:
        print('the size of the input image is wrong!')

    patch_size_hei = hei
    patch_size_wid = wid
    divisor = scale1*scale2

    for i in range(1, 301):
        if (hei % i == 0) and (i % divisor == 0):
            patch_size_hei = i

    for j in range(1, 301):
        if (wid % j == 0) and (j % divisor == 0):
            patch_size_wid = j

    return patch_size_hei, patch_size_wid


def ImgDividePad(img, border_size, patch_size_hei, patch_size_wid):
    if len(img.shape) == 3:
        hei, wid, cha = img.shape
    elif len(img.shape) == 2:
        (hei, wid), cha = img.shape, 1
    else:
        print('the size of the input image is wrong!')
    patch_size_hei_ = patch_size_hei + border_size*2
    patch_size_wid_ = patch_size_wid + border_size*2
    img_ext = cv2.copyMakeBorder(img, border_size, border_size, border_size, border_size, cv2.BORDER_REFLECT).reshape(hei + border_size*2, wid + border_size*2, cha)
    m = math.floor(hei/patch_size_hei)
    n = math.floor(wid/patch_size_wid)
    img_div = np.zeros((m*n, cha, patch_size_hei_, patch_size_wid_), dtype=np.float32)
    count = 0
    for i in range(m):
        for j in range(n):
            sub_img = img_ext[i*patch_size_hei:patch_size_hei_+i*patch_size_hei, j*patch_size_wid:patch_size_wid_+j*patch_size_wid, :]
            img_div[count, :, :, :] = sub_img.transpose(2, 0, 1)
            # mat2pymat(sub_img)
            count = count+1
    return img_div


def ImgDivide(img, patch_size_hei, patch_size_wid):
    if len(img.shape) == 3:
        hei, wid, cha = img.shape
    elif len(img.shape) == 2:
        (hei, wid), cha = img.shape, 1
    else:
        print('the size of the input image is wrong!')
    m = math.floor(hei/patch_size_hei)
    n = math.floor(wid/patch_size_wid)
    img_div = np.zeros((m*n, cha, patch_size_hei, patch_size_wid), dtype=np.float32)
    count = 0
    for i in range(m):
        for j in range(n):
            sub_img = img[i*patch_size_hei:patch_size_hei+i*patch_size_hei, j*patch_size_wid:patch_size_wid+j*patch_size_wid, :]
            img_div[count, :, :, :] = sub_img.transpose(2, 0, 1)
            # mat2pymat(sub_img)
            count = count+1
    return img_div


def ImgMergePad(img, border_size, hei_ori, wid_ori):
    if len(img.shape) == 4:
        num, cha, hei, wid = img.shape
    elif len(img.shape) == 3:
        (num, hei, wid), cha = img.shape, 1
    elif len(img.shape) == 2:
        (hei, wid), cha, num = img.shape, 1, 1
    else:
        print('the size of the input image is wrong!')

    img_deb = img[:, :, border_size: hei - border_size, border_size: wid - border_size]
    img_mer = mat_reshape(img_deb, hei_ori, wid_ori)
    return img_mer


def ImgMerge(img, hei_ori, wid_ori):
    img_mer = mat_reshape(img, hei_ori, wid_ori)
    return img_mer


def ReadGeoTiff(filename):
    dataset = gdal.Open(filename)
    if dataset == None:
        print(filename + "文件无法打开")
        return

    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    im_bands = dataset.RasterCount
    im_geotrans = dataset.GetGeoTransform()  # 获取仿射矩阵信息
    im_proj = dataset.GetProjection()  # 获取投影信息
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)
    return im_geotrans, im_proj, im_width, im_height, im_bands, im_data


def ReadTiff(filename):
    dataset = gdal.Open(filename)
    if dataset == None:
        print(filename + "文件无法打开")
        return

    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    im_bands = dataset.RasterCount
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)
    return im_data


def WriteGeoTiff(im_data, im_geotrans, im_proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        (im_height, im_width), im_bands = im_data.shape, 1
        im_data = np.array([im_data])
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, im_width, im_height, im_bands, datatype)
    # dataset = driver.Create(path, im_width, im_height, im_bands, datatype, options=["TILED=YES", "COMPRESS=LZW"])
    if (dataset != None):
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影

    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    dataset = None
