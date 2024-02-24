import torch.nn as nn
from kornia.filters import SpatialGradient
from torch import Tensor
import torch




class EdgeDetect(nn.Module):
    def __init__(self):
        super(EdgeDetect, self).__init__()
        self.spatial = SpatialGradient('diff')
        self.max_pool = nn.MaxPool2d(3, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        s = self.spatial(x)
        dx, dy = s[:, :, 0, :, :], s[:, :, 1, :, :]
        u = torch.sqrt(torch.pow(dx, 2) + torch.pow(dy, 2))
        y = self.max_pool(u)
        return y



class ConvConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvConv, self).__init__()

        self.conv_1 = nn.Conv2d(in_ch, out_ch, (3, 3), padding=(1, 1))
        self.conv_2 = nn.Conv2d(out_ch, out_ch, (3, 3), padding=(1, 1))
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.bn(x)
        x = self.conv_2(x)
        x = self.relu(x)
        x = self.bn(x)
        return x


class edge_enhencement_block(nn.Module):
    def __init__(self):
        super(edge_enhencement_block, self).__init__()

        self.conv_1 = ConvConv(1, 16)
        self.conv_2 = ConvConv(17, 32)
        self.conv_3 = nn.Conv2d(49, 1, (1, 1))
        self.ed = EdgeDetect()

    def forward(self, x):
        e = self.ed(x)


        e1 = self.conv_1(e)
        e2 = self.conv_2(torch.cat([e, e1], dim=1))
        e3 = self.conv_3(torch.cat([e, e1, e2], dim=1))

        attn_map = torch.sigmoid(e3)

        edge_map = attn_map * x


        return edge_map, attn_map








