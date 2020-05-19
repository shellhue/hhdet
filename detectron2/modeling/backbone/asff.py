import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init
import logging 
import math

from detectron2.layers import (
    Conv2d,
    FrozenBatchNorm2d,
    get_norm,
    CNNBlockBase,
)

class ASFF(nn.Module):
    def __init__(self, in_channels=[256, 256, 256], out_channels=[256, 256, 256], norm="BN"):
        super(ASFF, self).__init__()
        assert len(in_channels) == 3 and len(out_channels) == 3
        use_bias = norm == ""

        h_in_channel = in_channels[0]
        m_in_channel = in_channels[1]
        l_in_channel = in_channels[2]

        h_out_channel = out_channels[0]
        m_out_channel = out_channels[1]
        l_out_channel = out_channels[2]

        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.res_l_convs = [
            nn.Sequential(
                Conv2d(l_in_channel, h_in_channel, 1, 1),
                nn.UpsamplingNearest2d(scale_factor=4),
            ),
            nn.Sequential(
                Conv2d(l_in_channel, m_in_channel, 1, 1),
                nn.UpsamplingNearest2d(scale_factor=2),
            ),
            nn.Identity(),
        ]

        self.res_m_convs = [
            nn.Sequential(
                Conv2d(m_in_channel, h_in_channel, 1, 1),
                nn.UpsamplingNearest2d(scale_factor=2),
            ),
            nn.Identity(),
            Conv2d(m_in_channel, l_in_channel, 3, 2),
        ]

        self.res_h_convs = [
            nn.Identity(),
            Conv2d(h_in_channel, m_in_channel, 3, 2),
            nn.Sequential(
                nn.MaxPool2d(3, stride=2, padding=1),
                Conv2d(h_in_channel, l_in_channel, 3, 2),
            )
        ]

        self.output_convs = [
            Conv2d(h_in_channel, h_out_channel, 3, 1),
            Conv2d(m_in_channel, m_out_channel, 3, 1),
            Conv2d(l_in_channel, l_out_channel, 3, 1),
        ]

        compress_c = 8

        self.weights_h_convs = [
            Conv2d(h_in_channel, compress_c, 1, 1),
            Conv2d(m_in_channel, compress_c, 1, 1),
            Conv2d(l_in_channel, compress_c, 1, 1),
        ]
        self.weights_m_convs = [
            Conv2d(h_in_channel, compress_c, 1, 1),
            Conv2d(m_in_channel, compress_c, 1, 1),
            Conv2d(l_in_channel, compress_c, 1, 1),
        ]
        self.weights_l_convs = [
            Conv2d(h_in_channel, compress_c, 1, 1),
            Conv2d(m_in_channel, compress_c, 1, 1),
            Conv2d(l_in_channel, compress_c, 1, 1),
        ]

        self.weights_compress = [
            Conv2d(compress_c*3, 3, 1, 1),
            Conv2d(compress_c*3, 3, 1, 1),
            Conv2d(compress_c*3, 3, 1, 1),
        ]

        self.add_module("h2h_resize", res_h_convs[0])
        self.add_module("h2m_resize", res_h_convs[1])
        self.add_module("h2l_resize", res_h_convs[2])
        self.add_module("m2h_resize", res_m_convs[0])
        self.add_module("m2m_resize", res_m_convs[1])
        self.add_module("m2l_resize", res_m_convs[2])
        self.add_module("l2h_resize", res_l_convs[0])
        self.add_module("l2m_resize", res_l_convs[1])
        self.add_module("l2l_resize", res_l_convs[2])

        self.add_module("h_output", output_convs[0])
        self.add_module("m_output", output_convs[1])
        self.add_module("l_output", output_convs[2])

        self.add_module("h2h_weights", weights_h_convs[0])
        self.add_module("h2m_weights", weights_h_convs[1])
        self.add_module("h2l_weights", weights_h_convs[2])
        self.add_module("m2h_weights", weights_m_convs[0])
        self.add_module("m2m_weights", weights_m_convs[1])
        self.add_module("m2l_weights", weights_m_convs[2])
        self.add_module("l2h_weights", weights_l_convs[0])
        self.add_module("l2m_weights", weights_l_convs[1])
        self.add_module("l2l_weights", weights_l_convs[2])

        self.add_module("h_weights", weights_compress[0])
        self.add_module("m_weights", weights_compress[1])
        self.add_module("l_weights", weights_compress[2])

    def forward(self, x_h, x_m, x_l):
        results = []
        for i in range(3):
            x_h_resized = self.res_h_convs[i](x_h)
            x_m_resized = self.res_m_convs[i](x_m)
            x_l_resized = self.res_l_convs[i](x_l)

            x_h_weights = self.weights_h_convs[i](x_h_resized)
            x_m_weights = self.weights_m_convs[i](x_m_resized)
            x_l_weights = self.weights_l_convs[i](x_l_resized)

            x_weights = torch.cat((x_h_weights, x_m_weights, x_l_weights),1)
            x_weights = self.weights_compress[i](x_weights)
            x_weights = F.softmax(x_weights, dim=1)

            fused = x_h_resized * x_weights[:,0:1,:,:]+ x_m_resized * x_weights[:,1:2,:,:] + x_l_resized * x_weights[:,2:,:,:]
            x_out = self.output_convs(fused)
            results.append(x_out)

        return results