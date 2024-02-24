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

img1_path = ""
img2_path = ""
output_path = ""
model_path = ""
device = torch.device("cuda:0")






torch.backends.cudnn.benchmark = True
my_transforms = transforms.Compose([
    transforms.ToTensor(),
])
test_dataloader,dataset_size = fusion_datautils.load_fusion_dataset(img1_path, img2_path, transform=my_transforms, is_Train=False)
net = AC_fusion_model.Fusion_Model()
net = net.to(device)
net.eval()
model_state_dict = torch.load(model_path, map_location='cuda:0')
net.load_state_dict(model_state_dict)
net.to(device)
name_list = os.listdir(img1_path)



for idx, (img1, img2, name) in enumerate(test_dataloader):

    img1 = img1.to(device)
    img2 = img2.to(device)

    with torch.no_grad():
        fusion = net(img1, img2)


    fusion = (fusion - torch.min(fusion)) / (torch.max(fusion) - torch.min(fusion))
    fusion = fusion * 255
    fusion = np.array(fusion[0, :, :, :].detach().cpu().transpose(0, 1).transpose(1, 2))
    cv2.imwrite( output_path + name_list[idx] , fusion )








