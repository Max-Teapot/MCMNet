# 实现数据的加载类，杜绝对数据加载的烦恼，只适合pytorch框架
# coding=utf-8

import os
import random
import torch
import numpy as np
from PIL import Image, ImageFilter
from random import sample



class fusion_seg_DataSet(torch.utils.data.Dataset):

    # 重要的是split而不是mode
    def __init__(self,img1_path, img2_path, transform=None, is_Train=True, mul_scale=False, is_gray=False):
        self.img1_path = img1_path
        self.img2_path = img2_path
        self.transform = transform
        self.is_Train = is_Train
        self.crop_size_h = 384
        self.crop_size_w = 384
        self.mul_scale = mul_scale
        self.is_gray = is_gray

        self.img1_list = []
        self.img2_list = []
        self.masks = []
        # 训练过程，每次从列表里面随机抽取300张图片，测试过程就全部图片都获取
        if is_Train:
            # TNO数据集就不能这样
            self.name_list = sample(os.listdir(img1_path), 700)
            # self.name_list = os.listdir(img1_path)
        else:
            self.name_list = os.listdir(img1_path)
        for name in self.name_list:
            # 读取图片
            image1_path = img1_path + name
            assert os.path.isfile(image1_path)
            self.img1_list.append(image1_path)
            image2_path = img2_path + name
            assert os.path.isfile(image2_path)
            self.img2_list.append(image2_path)

        # 判断读取到的数据是否一致
        assert (len(self.img1_list) == len(self.img2_list))
        print('找到 {} 张图片，在 {} 目录下'.format(len(self.img1_list), img1_path))


    # 在getitem里面去读取图片，有利于数据增广
    def __getitem__(self, index):
        # ir在前，vi在后
        img1 = Image.open(self.img1_list[index])
        if self.is_gray:
            img2 = Image.open(self.img2_list[index]).convert("L")
        else:
            img2 = Image.open(self.img2_list[index])
        # img和mask要同步变化
        if self.is_Train:
            img1, img2 = self.train_transform(img1, img2)
        else:
            img1, img2 = self.test_transform(img1, img2)

        # 如果输入的transform不为空，那就使用transform
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        # 在训练过程中，只有前两个参数被用到。而在验证的时候，最后一个名字会用到
        return img1, img2, os.path.basename(self.img1_list[index])


    def __len__(self):
        return len(self.img1_list)


    # 处理验证集图片的手段
    def test_transform(self, img1, img2):
        # 不需要裁剪，和训练一样
        img1, img2 = self._img_transform(img1, img2)
        return img1, img2

    # 处理PIL的图片和mask
    def train_transform(self, img1, img2):
        # 随机裁剪
        if self.mul_scale:
            # 虽然裁剪之后的大小需要一样，但是可以通过缩放图片来达到不同裁剪的尺度。
            w, h = img1.size
            x1 = random.randint(0, w - self.crop_size_w)
            y1 = random.randint(0, h - self.crop_size_h)
            img1 = img1.crop((x1, y1, x1 + self.crop_size_w, y1 + self.crop_size_h))
            img2 = img2.crop((x1, y1, x1 + self.crop_size_w, y1 + self.crop_size_h))
        # 最终的变化，转化为numpy类型
        img1, img2 = self._img_transform(img1, img2)
        return img1, img2

    def _img_transform(self, img1, img2):
        return np.array(img1), np.array(img2)
















# 定义load函数，获取数据的dataloader
def load_fusion_dataset(img_path1, img_path2 ,batch_size=1, transform=None, is_Train=True, mul_scale=False, is_gray=False):
    dataset = fusion_seg_DataSet(img_path1, img_path2, transform, is_Train, mul_scale=mul_scale, is_gray=is_gray)
    data_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_Train, drop_last=True)
    return data_iter, len(dataset)























