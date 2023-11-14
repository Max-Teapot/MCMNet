import os
import time
from tqdm import tqdm
from Base_fusion_models import loss_fn
import cv2
import numpy as np
import torch.optim
import utils
from data_utils import fusion_datautils
import torchvision.transforms as transforms
from Unbais_fusion_model.inv_fusion_model import Inv_Fusion_Model,Sobelxy


# 单独训练可逆神经网络融合



model_path = "checkpoints/Edge_reinforce_Model/120_fusion_MSRS_SSIM.pth"
# 数据路径，先ir，后vi的顺序
train_img1_path = "datasets/RoadScene/ir/"
train_img2_path = "datasets/RoadScene/vi/"
final_path = "./final_result/Fusion_result/"
attn1_path = "./final_result/Attn_result1/"
attn2_path = "./final_result/Attn_result2/"
device = torch.device("cuda:0")


if os.path.exists(final_path) == False:
    os.makedirs(final_path)
if os.path.exists(attn1_path) == False:
    os.makedirs(attn1_path)
if os.path.exists(attn2_path) == False:
    os.makedirs(attn2_path)







# 一个totensor足以
#   1. 是将输入的数据shape H，W，C ——> C，H，W
#   2. 将所有数除以255，将数据归一化到 [0, 1]，这个归一化是很傻逼的归一化，它直接除以255
my_transforms = transforms.Compose([
    transforms.ToTensor(),
])
# 获取带label的数据集
test_loader,dataset_size = fusion_datautils.load_fusion_dataset(train_img1_path, train_img2_path, batch_size=1, transform=my_transforms, is_Train=False, is_gray=True)


# 创建模型
fusion_model = Inv_Fusion_Model(in_channel=2)
model_state_dict = torch.load(model_path, map_location='cuda:0')
fusion_model.load_state_dict(model_state_dict)
fusion_model = fusion_model.to(device)
fusion_model.eval()


# 求导算子
sobelxy = Sobelxy(channels=1)
sobelxy = sobelxy.to((device))






for idx,(img1, img2, name) in enumerate(test_loader):
    with torch.no_grad():
        # 获取图像和map
        img1 = img1.to(device)
        img2 = img2.to(device)


        # 不是简单的复制数据，而是求导
        img1_input = torch.cat([img1,img1], dim=1)
        img2_Y_input = torch.cat([img2,img2], dim=1)
        # 正向
        img1_feat = fusion_model(img1_input)
        img2_feat = fusion_model(img2_Y_input)
        # 反向
        fusion,feat_attn1,feat_attn2 = fusion_model(img1_feat, img2_feat, forward=False, ir_img = img1, vi_img = img2)
        # # 把复制的数据拿掉
        fusion = fusion[:,0:1,:,:]
        feat_attn1 = feat_attn1[:, 0:1, :, :]
        feat_attn2 = feat_attn2[:, 0:1, :, :]




        fusion = (fusion - torch.min(fusion)) / (torch.max(fusion) - torch.min(fusion))
        fusion = fusion * 255
        # 将获取到的图片保存下来
        fusion = np.array(fusion[0, :, :, :].detach().cpu().transpose(0, 1).transpose(1, 2))
        cv2.imwrite(final_path + name[0], fusion)
        # 将权重也保留
        feat_attn1 = (feat_attn1 - torch.min(feat_attn1)) / (torch.max(feat_attn1) - torch.min(feat_attn1))
        feat_attn1 = feat_attn1 * 255
        feat_attn1 = np.array(feat_attn1[0, :, :, :].detach().cpu().transpose(0, 1).transpose(1, 2))
        cv2.imwrite(attn1_path + name[0], feat_attn1)
        feat_attn2 = (feat_attn2 - torch.min(feat_attn2)) / (torch.max(feat_attn2) - torch.min(feat_attn2))
        feat_attn2 = feat_attn2 * 255
        feat_attn2 = np.array(feat_attn2[0, :, :, :].detach().cpu().transpose(0, 1).transpose(1, 2))
        cv2.imwrite(attn2_path + name[0], feat_attn2)

        # if idx+1 == 1:
        #     break












