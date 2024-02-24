import torch
from torch import nn
from . import cross_crise_attention
from . import Edge_Enhence_Block



class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1,dilation=1, groups=1, bias=False):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride,padding=padding, dilation=dilation,groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv(x)
        feat = self.bn(feat)
        feat = self.relu(feat)
        return feat


class StemBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(StemBlock, self).__init__()
        self.conv = ConvBNReLU(in_ch, in_ch*2, ks=3, stride=2)
        self.left = nn.Sequential(
            ConvBNReLU(in_ch*2, in_ch*2, 1, stride=1, padding=0),
            ConvBNReLU(in_ch*2, in_ch*2, 3, stride=2),
        )
        self.right = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        self.fuse = ConvBNReLU(in_ch*4, out_ch, 3, stride=1)

    def forward(self, x):
        feat = self.conv(x)
        feat_left = self.left(feat)
        feat_right = self.right(feat)
        feat = torch.cat([feat_left, feat_right], dim=1)
        feat = self.fuse(feat)
        return feat


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size= 1, bias=False)
        # self.bn_atten = torch.nn.BatchNorm2d(out_chan)
        self.init_weight()

    def forward(self, x):
        feat = self.conv(x)
        atten = torch.mean(feat, dim=(2, 3), keepdim=True)
        atten = self.conv_atten(atten)
        # atten = self.bn_atten(atten)
        atten = atten.sigmoid()
        out = torch.mul(feat, atten)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)






class Fusion_Block(nn.Module):
    def __init__(self, in_ch, edge_reinforce=False , device="cuda:0"):
        super(Fusion_Block, self).__init__()
        self.edge_reinforce = edge_reinforce
        self.stem1 = StemBlock(in_ch, in_ch)
        self.stem2 = StemBlock(in_ch, in_ch)
        self.cca1_1 = cross_crise_attention.CrissCrossAttention(in_ch, device)
        self.cca1_2 = cross_crise_attention.CrissCrossAttention(in_ch, device)
        self.cca2_1 = cross_crise_attention.CrissCrossAttention(in_ch, device)
        self.cca2_2 = cross_crise_attention.CrissCrossAttention(in_ch, device)
        self.conv_atten1 = nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False)
        self.conv_atten2 = nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.gate1 = nn.Parameter(torch.tensor(0.6), requires_grad=True)
        self.gate2 = nn.Parameter(torch.tensor(0.6), requires_grad=True)
        if self.edge_reinforce:
            self.edge_attn1 = Edge_Enhence_Block.edge_enhencement_block()
            self.edge_attn2 = Edge_Enhence_Block.edge_enhencement_block()



    def forward(self, feat1, feat2, ir_img, vi_img):
        feat1_copy = self.stem1(feat1)
        feat2_copy = self.stem2(feat2)
        if self.edge_reinforce:
            ir_edge_map,ir_attn_map = self.edge_attn1(ir_img)
            vi_edge_map,vi_attn_map = self.edge_attn2(vi_img)
            feat1 = feat1 + ir_edge_map
            feat2 = feat2 + vi_edge_map
        feat1_attn = self.cca1_1(feat1_copy, feat2_copy)
        feat1_attn = self.cca1_2(feat1_attn, feat2_copy)
        feat2_attn = self.cca2_1(feat2_copy, feat1_copy)
        feat2_attn = self.cca2_2(feat2_attn, feat1_copy)
        feat1_attn = self.conv_atten1(feat1_attn)
        feat2_attn = self.conv_atten2(feat2_attn)
        feat1_attn = feat1_attn.sigmoid()
        feat2_attn = feat2_attn.sigmoid()
        feat1_attn = self.upsample(feat1_attn)
        feat2_attn = self.upsample(feat2_attn)
        feat = feat1*self.gate1+(feat1*feat1_attn)*(1-self.gate1)  +  feat2*self.gate2+(feat2*feat2_attn)*(1-self.gate2)
        return feat,feat1_attn,feat2_attn










if __name__ == '__main__':
    x = torch.randn(1, 16, 160, 120, requires_grad=False)
    x = x.cuda()
    model = Fusion_Block(in_ch=16)
    model = model.cuda()

    total_num = sum(p.numel() for p in model.parameters())
    print('Total number of parameters : %.6f M' % (total_num / 1e6))








