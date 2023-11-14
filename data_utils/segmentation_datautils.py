
import os
import random
import numpy as np

from PIL import Image, ImageFilter
import torch



class Seg_DataSet(torch.utils.data.Dataset):

    # 重要的是split而不是mode
    def __init__(self,img_path, label_path, num_classes, transform=None, is_Train=True, mul_scale=False):
        self.img_path = img_path
        self.label_path = label_path
        self.num_classes = num_classes
        self.transform = transform
        self.is_Train = is_Train
        self.crop_size_h = 480
        self.crop_size_w = 640
        self.mul_scale = mul_scale


        self.images = []
        self.masks = []
        self.name_list = os.listdir(img_path)
        for name in self.name_list:
            # 读取图片
            image_path = img_path + name
            assert os.path.isfile(image_path)
            self.images.append(image_path)
            # 读取标签
            mask_path = label_path + name
            assert os.path.isfile(mask_path)
            self.masks.append(mask_path)

        # 判断读取到的数据是否一致
        assert (len(self.images) == len(self.masks))
        print('找到 {} 张图片，在 {} 目录下'.format(len(self.images), img_path))


    # 在getitem里面去读取图片，有利于数据增广
    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        # P模式的图其实是彩图，表示通过调色板，存储256中颜色，但实际上就是单通道
        # 这里的mask的最大值还是255，也就是还没有映射到0~20
        mask = Image.open(self.masks[index])
        # img和mask要同步变化
        if self.is_Train:
            img, mask = self.train_transform(img, mask)
        else:
            img, mask = self.test_transform(img, mask)

        # 如果输入的transform不为空，那就使用transform
        if self.transform is not None:
            img = self.transform(img)

        # 在训练过程中，只有前两个参数被用到。而在验证的时候，最后一个名字会用到
        return img, mask, os.path.basename(self.images[index])


    def __len__(self):
        return len(self.images)



    # 处理验证集图片的手段
    def test_transform(self, img, mask):
        # 不需要裁剪，和训练一样
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask


    # 处理PIL的图片和mask
    def train_transform(self, img, mask):
        # 随机旋转
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        # 随机裁剪
        if self.mul_scale:
            # 虽然裁剪之后的大小需要一样，但是可以通过缩放图片来达到不同裁剪的尺度。
            short_size = random.randint(480, int(480 * 1.5))
            w, h = img.size
            if h > w:
                ow = short_size
                oh = int(1.0 * h * ow / w)
            else:
                oh = short_size
                ow = int(1.0 * w * oh / h)
            # 一个采取双线性插值，一个采取临近插值
            img = img.resize((ow, oh), Image.BILINEAR)
            mask = mask.resize((ow, oh), Image.NEAREST)
            # 进行裁剪
            w, h = img.size
            x1 = random.randint(0, w - self.crop_size_w)
            y1 = random.randint(0, h - self.crop_size_h)
            img = img.crop((x1, y1, x1 + self.crop_size_w, y1 + self.crop_size_h))
            mask = mask.crop((x1, y1, x1 + self.crop_size_w, y1 + self.crop_size_h))
        # 最终的变化，转化为numpy类型
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _img_transform(self, img):
        return np.array(img)

    # 这个方法被不同的类重写了，所以不会进入这里执行，而是进入重写的方法
    def _mask_transform(self, mask):
        result = np.array(mask, dtype=np.int64)
        return result

    @property
    def num_class(self):
        """Number of categories."""
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        return 0











# 读取数据集的方法，要传入一个参数
def read_Seg_data(img_path, label_path, num_classes, batch_size, transform, is_Train=True, mul_scale=False):
    dataset = Seg_DataSet(img_path, label_path , num_classes, transform, is_Train, mul_scale=mul_scale)
    data_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_Train, drop_last=True)
    return data_iter, len(dataset)





