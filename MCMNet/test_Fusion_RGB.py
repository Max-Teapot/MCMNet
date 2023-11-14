import os
import torch.optim
import utils
from utils import save_one_img_L, save_one_img_RGB
from data_utils import fusion_datautils
import torchvision.transforms as transforms
from Unbais_fusion_model.inv_fusion_model import Inv_Fusion_Model


# 单独训练可逆神经网络融合



model_path = "checkpoints/unbias_fusion_model/170_fusion.pth"
# 数据路径，先ir，后vi的顺序
train_img1_path = "datasets/RoadScene/ir/"
train_img2_path = "datasets/RoadScene/vi/"
final_path = "./final_result/Fusion_result/"
attn1_path = "./final_result/Attn_result1/"
attn2_path = "./final_result/Attn_result2/"
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





# 一个totensor足以
#   1. 是将输入的数据shape H，W，C ——> C，H，W
#   2. 将所有数除以255，将数据归一化到 [0, 1]，这个归一化是很傻逼的归一化，它直接除以255
my_transforms = transforms.Compose([
    transforms.ToTensor(),
])
# 获取带label的数据集
test_loader,dataset_size = fusion_datautils.load_fusion_dataset(train_img1_path, train_img2_path, batch_size=1, transform=my_transforms, is_Train=False)

# 创建模型
fusion_model = Inv_Fusion_Model(in_channel=2)
model_state_dict = torch.load(model_path, map_location=device)
fusion_model.load_state_dict(model_state_dict)
fusion_model = fusion_model.to(device)
fusion_model.eval()









for idx,(img1, img2, name) in enumerate(test_loader):
    print(name)
    with torch.no_grad():
        # 获取图像和map
        img1 = img1.to(device)
        img2_YCrCb = utils.RGB2YCrCb(img2, device)
        img2_Y, img2_Cb, img2_Cr = img2_YCrCb[:, 0:1, :, :], img2_YCrCb[:, 1:2, :, :], img2_YCrCb[:, 2:, :, :]
        img2_Y, img2_Cr, img2_Cb = img2_Y.to(device), img2_Cr.to(device), img2_Cb.to(device)

        # 不是简单的复制数据，而是求导
        img1_input = torch.cat([img1,img1], dim=1)
        img2_Y_input = torch.cat([img2_Y,img2_Y], dim=1)
        # 正向
        img1_feat = fusion_model(img1_input)
        img2_feat = fusion_model(img2_Y_input)
        # 反向
        fusion,feat_attn1,feat_attn2 = fusion_model(img1_feat, img2_feat, forward=False, ir_img = img1, vi_img = img2_Y)

        fusion_save_list = []


        # 判断是否为测试细节泄露
        if not test_leakage:
            fusion_save = fusion[:, 0:1, :, :]
            save_one_img_RGB(fusion_save, img2_Cr, img2_Cb, final_path + name[0], device)
            if save_attention:
                # 需要保留attn图
                feat_attn1 = feat_attn1[:, 0:1, :, :]
                feat_attn2 = feat_attn2[:, 0:1, :, :]
                save_one_img_L(feat_attn1, attn1_path + name[0])
                save_one_img_L(feat_attn2, attn2_path + name[0])
        else:
            # 需要测试内容泄露
            fusion_save = fusion[:, 0:1, :, :]
            save_one_img_RGB(fusion_save, img2_Cr, img2_Cb, "./final_result/Detail_leak_result/Round1/" + name[0], device)
            for i in range(2, 21):
                fusion = fusion[:, 0:1, :, :]
                fusion = torch.cat([img2_Y, fusion], dim=1)
                fusion = fusion_model(fusion)
                # 反向
                fusion,_,_ = fusion_model(img1_feat, fusion, forward=False)
                # 把复制的数据拿掉
                fusion_save = fusion[:, 0:1, :, :]
                save_one_img_RGB(fusion_save, img2_Cr, img2_Cb, "./final_result/Detail_leak_result/Round" + str(i) + "/" + name[0], device)

    # if idx == 149:
    #     break


















