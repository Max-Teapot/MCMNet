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





model_path = ""
train_img1_path = ""
train_img2_path = ""
final_path = ""
attn1_path = ""
attn2_path = ""
device = torch.device("cuda:0")


if os.path.exists(final_path) == False:
    os.makedirs(final_path)
if os.path.exists(attn1_path) == False:
    os.makedirs(attn1_path)
if os.path.exists(attn2_path) == False:
    os.makedirs(attn2_path)








my_transforms = transforms.Compose([
    transforms.ToTensor(),
])
test_loader,dataset_size = fusion_datautils.load_fusion_dataset(train_img1_path, train_img2_path, batch_size=1, transform=my_transforms, is_Train=False, is_gray=True)


fusion_model = Inv_Fusion_Model(in_channel=2)
model_state_dict = torch.load(model_path, map_location='cuda:0')
fusion_model.load_state_dict(model_state_dict)
fusion_model = fusion_model.to(device)
fusion_model.eval()


sobelxy = Sobelxy(channels=1)
sobelxy = sobelxy.to((device))






for idx,(img1, img2, name) in enumerate(test_loader):
    with torch.no_grad():
        img1 = img1.to(device)
        img2 = img2.to(device)


        img1_input = torch.cat([img1,img1], dim=1)
        img2_Y_input = torch.cat([img2,img2], dim=1)
        img1_feat = fusion_model(img1_input)
        img2_feat = fusion_model(img2_Y_input)
        fusion,feat_attn1,feat_attn2 = fusion_model(img1_feat, img2_feat, forward=False, ir_img = img1, vi_img = img2)
        fusion = fusion[:,0:1,:,:]
        feat_attn1 = feat_attn1[:, 0:1, :, :]
        feat_attn2 = feat_attn2[:, 0:1, :, :]




        fusion = (fusion - torch.min(fusion)) / (torch.max(fusion) - torch.min(fusion))
        fusion = fusion * 255
        fusion = np.array(fusion[0, :, :, :].detach().cpu().transpose(0, 1).transpose(1, 2))
        cv2.imwrite(final_path + name[0], fusion)
        feat_attn1 = (feat_attn1 - torch.min(feat_attn1)) / (torch.max(feat_attn1) - torch.min(feat_attn1))
        feat_attn1 = feat_attn1 * 255
        feat_attn1 = np.array(feat_attn1[0, :, :, :].detach().cpu().transpose(0, 1).transpose(1, 2))
        cv2.imwrite(attn1_path + name[0], feat_attn1)
        feat_attn2 = (feat_attn2 - torch.min(feat_attn2)) / (torch.max(feat_attn2) - torch.min(feat_attn2))
        feat_attn2 = feat_attn2 * 255
        feat_attn2 = np.array(feat_attn2[0, :, :, :].detach().cpu().transpose(0, 1).transpose(1, 2))
        cv2.imwrite(attn2_path + name[0], feat_attn2)













