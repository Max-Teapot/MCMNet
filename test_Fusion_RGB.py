import sys

import cv2

import utils
from Base_fusion_models import AC_fusion_model

sys.path.append("..")
import os
import torch.backends.cudnn
import torch.utils.data
from data_utils import fusion_datautils
import numpy as np
import torchvision.transforms as transforms
from utils import save_one_img_L, save_one_img_RGB



model_path = ""
img1_path = ""
img2_path = ""
final_path = ""
device = torch.device("cuda:0")
test_leakage = True



if not test_leakage:
    os.makedirs(final_path)
else:
    for i in range(1,21):
        dir = "./final_result/Detail_leak_result/Round" + str(i)
        if os.path.exists(dir) == False:
            os.makedirs(dir)




torch.backends.cudnn.benchmark = True
my_transforms = transforms.Compose([
    transforms.ToTensor(),
])
test_dataloader,_ = fusion_datautils.load_fusion_dataset(img1_path, img2_path, transform=my_transforms, is_Train=False)
net = AC_fusion_model.Fusion_Model()
net = net.to(device)
net.eval()
model_state_dict = torch.load(model_path, map_location='cuda:0')
net.load_state_dict(model_state_dict)
net.to(device)
name_list = os.listdir(img1_path)



for idx, (img1, img2, name) in enumerate(test_dataloader):

    img1 = img1.to(device)
    img2_YCbCr = utils.RGB2YCrCb(img2, device)
    img2_Y, img2_Cb, img2_Cr = img2_YCbCr[:, 0:1, :, :], img2_YCbCr[:, 1:2, :, :], img2_YCbCr[:, 2:, :, :]
    img2_Y, img2_Cb, img2_Cr = img2_Y.to(device), img2_Cb.to(device), img2_Cr.to(device)


    with torch.no_grad():
        if not test_leakage:
            fusion = net(img1, img2_Y)
            save_one_img_RGB(fusion, img2_Cb, img2_Cr, final_path + name[0], device)
        else:
            fusion = net(img1, img2_Y)
            save_one_img_RGB(fusion, img2_Cb, img2_Cr, "./final_result/Detail_leak_result/Round1/" + name[0], device)
            for i in range(2, 21):
                fusion = net(img1, fusion)
                save_one_img_RGB(fusion, img2_Cb, img2_Cr, "./final_result/Detail_leak_result/Round" + str(i) + "/" + name[0], device)






    if idx == 200:
        break






