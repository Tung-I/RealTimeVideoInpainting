import torch
import torch.nn as nn
import torch.nn.functional as F

from model.net.base_net import BaseNet


class STBNet(BaseNet):
    """
    """
    def __init__(self, in_channels, out_channels, num_features=[32, 64, 128]):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_features = num_features

        self.in_block = _InBlock(in_channels, num_features[0])
        self.down_block1 = _DownBlock(num_features[0], num_features[1])
        self.down_block2 = _DownBlock(num_features[1], num_features[2])

        self.up_block1 = _UpBlock(num_features[2], num_features[1])
        self.up_block2 = _UpBlock(num_features[1], num_features[0])
        self.out_block = _OutBlock(num_features[0], out_channels)

    def forward(self, input):
        # Space to batch
        N, C, H, W = input.size()
        n_split = 4
        x = input.view(N, C, H//n_split, n_split, W//n_split, n_split)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H', W')
        x = x.view(N*(n_split**2), C, H//n_split, W//n_split)

        # Encoder
        features1 = self.in_block(x)
        features2 = self.down_block1(features1)
        x = self.down_block2(features2)

        # Decoder
        x = self.up_block1(x, features2)
        x = self.up_block2(x, features1)

        # Batch to space
        N, C, H, W = x.size()
        x = x.view(N//(n_split**2), n_split, n_split, C, H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # (N, C, H', bs, W', bs)
        x = x.view(N//(n_split**2), C, H*n_split, W*n_split)

        output = self.out_block(x)
        return output


class _InBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.add_module('conv1', nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        self.add_module('norm1', nn.BatchNorm2d(out_channels))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        self.add_module('norm2', nn.BatchNorm2d(out_channels))
        self.add_module('relu2', nn.ReLU(inplace=True))


class _DownBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.add_module('pool', nn.MaxPool2d(2))
        self.add_module('conv1', nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        self.add_module('norm1', nn.BatchNorm2d(out_channels))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        self.add_module('norm2', nn.BatchNorm2d(out_channels))
        self.add_module('relu2', nn.ReLU(inplace=True))


class _UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU(inplace=True))

    def forward(self, input, features):
        input = self.deconv(input)

        h_diff = features.size(2) - input.size(2)
        w_diff = features.size(3) - input.size(3)
        input = F.pad(input, (w_diff // 2, w_diff - w_diff//2,
                              h_diff // 2, h_diff - h_diff//2))

        output = self.conv(torch.cat([input, features], dim=1))
        return output


class _OutBlock(nn.Conv2d):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, kernel_size=1)