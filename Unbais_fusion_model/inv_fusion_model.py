import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from scipy import linalg as la
import numpy as np
from . import invblock
from . import feature_fusion_model
# import invblock
# import feature_fusion_model


logabs = lambda x: torch.log(torch.abs(x))



class HaarDownsampling(nn.Module):
    def __init__(self, channel_in):
        super(HaarDownsampling, self).__init__()
        self.channel_in = channel_in

        self.haar_weights = torch.ones(4, 1, 2, 2)

        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.channel_in, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

    def forward(self, x, rev=False):
        if not rev:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]

            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.channel_in) / 4.0
            out = out.reshape([x.shape[0], self.channel_in, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)

            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2])
            return out
        else:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]

            out = x.reshape([x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]])
            return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups = self.channel_in)











class Sobelxy(nn.Module):
    def __init__(self,channels, kernel_size=3, padding=1, stride=1, dilation=1):
        super(Sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        self.convx=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.convy=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convy.weight.data.copy_(torch.from_numpy(sobel_filter.T))
    def forward(self, x):
        sobelx = self.convx(x)
        sobely = self.convy(x)
        x=torch.abs(sobelx) + torch.abs(sobely)
        return x







class ActNorm(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))
        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (flatten.mean(1).unsqueeze(1).unsqueeze(2).unsqueeze(3).permute(1, 0, 2, 3))
            std = (flatten.std(1).unsqueeze(1).unsqueeze(2).unsqueeze(3).permute(1, 0, 2, 3))
            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input):
        _, _, height, width = input.shape
        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)
        return self.scale * (input + self.loc)

    def reverse(self, output):
        return output / self.scale - self.loc



class InvConv2d(nn.Module):
    def __init__(self, in_channel, out_channel=None):
        super().__init__()
        if out_channel is None:
            out_channel = in_channel
        weight = torch.randn(in_channel, out_channel)
        q, _ = torch.linalg.qr(weight)
        weight = q.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(weight)

    def forward(self, input):
        _, _, height, width = input.shape
        out = F.conv2d(input, self.weight)
        return out

    def reverse(self, output):
        return F.conv2d(output, self.weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))





class InvConv2dLU(nn.Module):
    def __init__(self, in_channel, out_channel=None):
        super().__init__()
        if out_channel is None:
            out_channel = in_channel
        weight = np.random.randn(in_channel, out_channel)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T
        w_p = torch.from_numpy(w_p.copy())
        w_l = torch.from_numpy(w_l.copy())
        w_s = torch.from_numpy(w_s.copy())
        w_u = torch.from_numpy(w_u.copy())
        self.register_buffer('w_p', w_p)
        self.register_buffer('u_mask', torch.from_numpy(u_mask))
        self.register_buffer('l_mask', torch.from_numpy(l_mask))
        self.register_buffer('s_sign', torch.sign(w_s))
        self.register_buffer('l_eye', torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(logabs(w_s))
        self.w_u = nn.Parameter(w_u)

    def forward(self, input):
        _, _, height, width = input.shape
        weight = self.calc_weight()
        out = F.conv2d(input, weight)
        return out

    def calc_weight(self):
        weight = (self.w_p @ (self.w_l * self.l_mask + self.l_eye) @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s))))
        return weight.unsqueeze(2).unsqueeze(3)

    def reverse(self, output):
        weight = self.calc_weight()
        return F.conv2d(output, weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))







class Flow(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.actnorm = ActNorm(in_channel)
        self.invconv = InvConv2dLU(in_channel)
        self.coupling = invblock.INV_block_affine(in_channel)

    def forward(self, input):
        input = self.actnorm(input)
        input = self.invconv(input)
        input = self.coupling(input)
        return input

    def reverse(self, input):
        input = self.coupling.reverse(input)
        input = self.invconv.reverse(input)
        input = self.actnorm.reverse(input)
        return input







class Block(nn.Module):
    def __init__(self, in_channel, n_flow):
        super().__init__()
        squeeze_dim = in_channel
        self.flows = nn.ModuleList()
        for i in range(n_flow):
            self.flows.append(Flow(squeeze_dim))

    def forward(self, input):
        for flow in self.flows:
            input = flow(input)
        return input

    def reverse(self, output):
        for flow in self.flows[::-1]:
            output = flow.reverse(output)
        return output





class Inv_Fusion_Model(nn.Module):
    def __init__(self, in_channel=1, device="cuda:0", n_flow=4, n_block=2):
        super().__init__()
        self.blocks = nn.ModuleList()
        n_channel = in_channel
        for i in range(n_block - 1):
            self.blocks.append(Block(n_channel, n_flow))
        self.blocks.append(Block(n_channel, n_flow))
        self.fusion_block = feature_fusion_model.Fusion_Block(in_ch=2, device=device)


    def forward(self, feat1, feat2=None, forward=True,ir_img = None, vi_img=None, only_inn=False):
        if forward:
            return self._forward(feat1)
        else:
            return self._reverse( feat1, feat2, ir_img, vi_img, only_inn)

    def _forward(self, feat1):
        z = feat1
        for block in self.blocks:
            z = block(z)
        return z

    def _reverse(self, feat1, feat2, ir_img, vi_img, only_inn=False):
        if only_inn:
            out = feat1+feat2
            feat_attn1=torch.randn(1)
            feat_attn2=torch.randn(1)
        else:
            out,feat_attn1,feat_attn2 = self.fusion_block(feat1, feat2, ir_img, vi_img)
        for i, block in enumerate(self.blocks[::-1]):
            out = block.reverse(out)
        return out,feat_attn1,feat_attn2



if __name__ == '__main__':
    x = torch.randn(1, 2, 64, 64, requires_grad=False)
    x = x.cuda()
    model = Inv_Fusion_Model(in_channel=2)
    model = model.cuda()

    total_num = sum(p.numel() for p in model.parameters())
    print('Total number of parameters : %.6f M' % (total_num / 1e6))










