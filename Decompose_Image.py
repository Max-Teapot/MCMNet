# coding=utf-8
import numpy as np
import cv2


def boxfilter(img, r):
    (rows, cols) = img.shape
    imDst = np.zeros_like(img)
    imCum = np.cumsum(img, 0)
    imDst[0 : r+1, :] = imCum[r : 2*r+1, :]
    imDst[r+1 : rows-r, :] = imCum[2*r+1 : rows, :] - imCum[0 : rows-2*r-1, :]
    imDst[rows-r: rows, :] = np.tile(imCum[rows-1, :], [r, 1]) - imCum[rows-2*r-1 : rows-r-1, :]
    imCum = np.cumsum(imDst, 1)
    imDst[:, 0 : r+1] = imCum[:, r : 2*r+1]
    imDst[:, r+1 : cols-r] = imCum[:, 2*r+1 : cols] - imCum[:, 0 : cols-2*r-1]
    imDst[:, cols-r: cols] = np.tile(imCum[:, cols-1], [r, 1]).T - imCum[:, cols-2*r-1 : cols-r-1]
    return imDst


def guidedfilter(I, p, r, eps):
    (rows, cols) = I.shape
    N = boxfilter(np.ones([rows, cols]), r)
    meanI = boxfilter(I, r) / N
    meanP = boxfilter(p, r) / N
    meanIp = boxfilter(I * p, r) / N
    covIp = meanIp - meanI * meanP
    meanII = boxfilter(I * I, r) / N
    varI = meanII - meanI * meanI
    a = covIp / (varI + eps)
    b = meanP - a * meanI
    meanA = boxfilter(a, r) / N
    meanB = boxfilter(b, r) / N
    q = meanA * I + meanB
    return q



def RollingGuidanceFilter_Guided(I, sigma_s, sigma_r, iteration):
    kernal_size = sigma_s * 6 + 1
    res = cv2.GaussianBlur(I, [kernal_size,kernal_size], sigma_s)
    for i in range(iteration):
        G = res
        res = guidedfilter(G, I, sigma_s, sigma_r**2)
    return res





def Decompose(Image):
    nLevel = 4
    sigma_s = 2
    sigma_r = 0.05
    iteration = 4

    H,W = Image.shape
    G = np.ones((nLevel+1, H,W), dtype=float)
    Detail_layers = np.ones((nLevel, H,W), dtype=float)
    Detail_layer = np.zeros(( H,W ), dtype=float)

    G[0,:,:] = Image
    for i in range(1, nLevel):
        G[i, :, :] = RollingGuidanceFilter_Guided(G[i-1, :, :],sigma_s,sigma_r,iteration);
        Detail_layers[i-1, :, :] = G[i-1, :, :] - G[i, :, :]
        sigma_s = 2 * sigma_s;

    sigma_s = 2;
    G[4, :, :] = cv2.GaussianBlur(G[3, :, :], [sigma_s*6+1, sigma_s*6+1], sigma_s)
    Detail_layers[3, :, :] = G[3, :, :] - G[4, :, :]

    Basic_layer = G[4, :, :]
    for i in range(4):
        Detail_layer = Detail_layer + Detail_layers[i , :, :]
    return Detail_layer, Basic_layer



















