# 实现数据的加载类，杜绝对数据加载的烦恼，只适合pytorch框架
# coding=utf-8
import os
import random

import torch
import numpy as np
from PIL import Image, ImageFilter








class fusion_seg_DataSet(torch.utils.data.Dataset):

    # 重要的是split而不是mode
    def __init__(self,img1_path, img2_path, label_path, transform=None, is_Train=True, mul_scale=False):
        self.img1_path = img1_path
        self.img2_path = img2_path
        self.label_path = label_path
        self.transform = transform
        self.is_Train = is_Train
        self.crop_size_h = 480
        self.crop_size_w = 640
        self.mul_scale = mul_scale

        self.img1_list = []
        self.img2_list = []
        self.masks = []
        self.name_list = os.listdir(img1_path)
        for name in self.name_list:
            # 读取图片
            image1_path = img1_path + name
            assert os.path.isfile(image1_path)
            self.img1_list.append(image1_path)
            image2_path = img2_path + name
            assert os.path.isfile(image2_path)
            self.img2_list.append(image2_path)
            # 读取标签
            mask_path = label_path + name
            assert os.path.isfile(mask_path)
            self.masks.append(mask_path)

        # 判断读取到的数据是否一致
        assert (len(self.img1_list) == len(self.masks))
        print('找到 {} 张图片，在 {} 目录下'.format(len(self.img1_list), img1_path))


    # 在getitem里面去读取图片，有利于数据增广
    def __getitem__(self, index):
        # ir在前，vi在后
        img1 = Image.open(self.img1_list[index]).convert('L')
        img2 = Image.open(self.img2_list[index]).convert('RGB')
        # P模式的图其实是彩图，表示通过调色板，存储256中颜色，但实际上就是单通道
        # 这里的mask的最大值还是255，也就是还没有映射到0~20
        label = Image.open(self.masks[index])
        # img和mask要同步变化
        if self.is_Train:
            img1, img2, label = self.train_transform(img1, img2, label)
        else:
            img1, img2, label = self.test_transform(img1, img2, label)

        # 如果输入的transform不为空，那就使用transform
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        # 在训练过程中，只有前两个参数被用到。而在验证的时候，最后一个名字会用到
        return img1, img2, label, os.path.basename(self.img1_list[index])


    def __len__(self):
        return len(self.img1_list)


    # 处理验证集图片的手段
    def test_transform(self, img1, img2, label):
        # 不需要裁剪，和训练一样
        img1, img2 = self._img_transform(img1, img2)
        label = self._mask_transform(label)
        return img1, img2, label

    # 处理PIL的图片和mask
    def train_transform(self, img1, img2, label):
        # 随机旋转
        if random.random() < 0.5:
            img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
            img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        # 随机裁剪
        if self.mul_scale:
            # 虽然裁剪之后的大小需要一样，但是可以通过缩放图片来达到不同裁剪的尺度。
            short_size = random.randint(480, int(480 * 2.0))
            w, h = img1.size
            if h > w:
                ow = short_size
                oh = int(1.0 * h * ow / w)
            else:
                oh = short_size
                ow = int(1.0 * w * oh / h)
            # 一个采取双线性插值，一个采取临近插值
            img1 = img1.resize((ow, oh), Image.BILINEAR)
            img2 = img2.resize((ow, oh), Image.BILINEAR)
            label = label.resize((ow, oh), Image.NEAREST)
            # 进行裁剪
            w, h = img1.size
            x1 = random.randint(0, w - self.crop_size_w)
            y1 = random.randint(0, h - self.crop_size_h)
            img1 = img1.crop((x1, y1, x1 + self.crop_size_w, y1 + self.crop_size_h))
            img2 = img2.crop((x1, y1, x1 + self.crop_size_w, y1 + self.crop_size_h))
            label = label.crop((x1, y1, x1 + self.crop_size_w, y1 + self.crop_size_h))
        # 随机高斯模糊
        if random.random() < 0.3:
            img1 = img1.filter(ImageFilter.GaussianBlur(radius=random.random()))
            img2 = img2.filter(ImageFilter.GaussianBlur(radius=random.random()))
        # 最终的变化，转化为numpy类型
        img1, img2 = self._img_transform(img1, img2)
        label = self._mask_transform(label)
        return img1, img2, label

    def _img_transform(self, img1, img2):
        return np.array(img1), np.array(img2)

    # 这个方法被不同的类重写了，所以不会进入这里执行，而是进入重写的方法
    def _mask_transform(self, label):
        result = np.array(label, dtype=np.int64)
        return result













# 读取数据集的方法，要传入一个参数
def read_fusion_seg_data(img1_path, img2_path, label_path, batch_size, transform, is_Train=True, mul_scale=False):
    dataset = fusion_seg_DataSet(img1_path, img2_path, label_path, transform, is_Train, mul_scale=mul_scale)
    data_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_Train, drop_last=True)
    return data_iter, len(dataset)
























