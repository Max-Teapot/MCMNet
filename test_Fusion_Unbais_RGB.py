import os
import torch.optim
import utils
from utils import save_one_img_L, save_one_img_RGB
from data_utils import fusion_datautils
import torchvision.transforms as transforms
from Unbais_fusion_model.inv_fusion_model import Inv_Fusion_Model





model_path = ""
train_img1_path = ""
train_img2_path = ""
final_path = ""
attn1_path = ""
attn2_path = ""
device = torch.device("cuda:0")
test_leakage = False
save_attention = False




if not test_leakage:
    if os.path.exists(final_path) == False:
        os.makedirs(final_path)
    if save_attention:
        if os.path.exists(attn1_path) == False:
            os.makedirs(attn1_path)
        if os.path.exists(attn2_path) == False:
            os.makedirs(attn2_path)
else:
    for i in range(1,21):
        dir = "./final_result/Detail_leak_result/Round" + str(i)
        if os.path.exists(dir) == False:
            os.makedirs(dir)






my_transforms = transforms.Compose([
    transforms.ToTensor(),
])
test_loader,dataset_size = fusion_datautils.load_fusion_dataset(train_img1_path, train_img2_path, batch_size=1, transform=my_transforms, is_Train=False)

fusion_model = Inv_Fusion_Model(in_channel=2)
model_state_dict = torch.load(model_path, map_location=device)
fusion_model.load_state_dict(model_state_dict)
fusion_model = fusion_model.to(device)
fusion_model.eval()









for idx,(img1, img2, name) in enumerate(test_loader):
    print(name)
    with torch.no_grad():
        img1 = img1.to(device)
        img2_YCrCb = utils.RGB2YCrCb(img2, device)
        img2_Y, img2_Cb, img2_Cr = img2_YCrCb[:, 0:1, :, :], img2_YCrCb[:, 1:2, :, :], img2_YCrCb[:, 2:, :, :]
        img2_Y, img2_Cr, img2_Cb = img2_Y.to(device), img2_Cr.to(device), img2_Cb.to(device)

        img1_input = torch.cat([img1,img1], dim=1)
        img2_Y_input = torch.cat([img2_Y,img2_Y], dim=1)
        img1_feat = fusion_model(img1_input)
        img2_feat = fusion_model(img2_Y_input)
        fusion,feat_attn1,feat_attn2 = fusion_model(img1_feat, img2_feat, forward=False, ir_img = img1, vi_img = img2_Y)

        fusion_save_list = []


        if not test_leakage:
            fusion_save = fusion[:, 0:1, :, :]
            save_one_img_RGB(fusion_save, img2_Cr, img2_Cb, final_path + name[0], device)
            if save_attention:
                feat_attn1 = feat_attn1[:, 0:1, :, :]
                feat_attn2 = feat_attn2[:, 0:1, :, :]
                save_one_img_L(feat_attn1, attn1_path + name[0])
                save_one_img_L(feat_attn2, attn2_path + name[0])
        else:
            fusion_save = fusion[:, 0:1, :, :]
            save_one_img_RGB(fusion_save, img2_Cr, img2_Cb, "./final_result/Detail_leak_result/Round1/" + name[0], device)
            for i in range(2, 21):
                fusion = fusion[:, 0:1, :, :]
                fusion = torch.cat([img2_Y, fusion], dim=1)
                fusion = fusion_model(fusion)
                fusion,_,_ = fusion_model(img1_feat, fusion, forward=False)
                fusion_save = fusion[:, 0:1, :, :]
                save_one_img_RGB(fusion_save, img2_Cr, img2_Cb, "./final_result/Detail_leak_result/Round" + str(i) + "/" + name[0], device)



















