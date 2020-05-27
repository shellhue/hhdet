import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init
import logging 
import math

from detectron2.modeling.backbone.darknet import Conv2dBNLeakyReLU

from detectron2.layers import (
    Conv2d,
    FrozenBatchNorm2d,
    get_norm,
    CNNBlockBase,
)

""" 
This module contains all the FPG(Feature Pyramid Generator) for FPN(Feature Pyramid Networks)
"""

__all__ = [
    "Yolov3FPG", 
    "RetinaFPG",
    "BiFPG"
    ]

class PANetFF(nn.Module):
    """
    Feature fusion from bottom up direction used in PANet(https://arxiv.org/abs/1803.01534)
    """
    def __init__(self, in_channels=[256, 256, 256], out_channels=[256, 256, 256], norm="BN"):
        super(PANetFF, self).__init__()
        assert len(in_channels) == len(out_channels)
        assert in_channels[0] == out_channels[0]

        self.in_channels = in_channels
        self.out_channels = out_channels

        relu = nn.ReLU(inplace=True)

        bottom_up_convs = []
        output_convs = []

        for i in range(1, len(in_channels)):
            bp_conv = Conv2d(out_channels[i-1], in_channels[i], kernel_size=3, stride=2, padding=1, norm=get_norm(norm, in_channels[i]), activation=relu)
            o_conv = Conv2d(in_channels[i], out_channels[i], kernel_size=3, stride=1, padding=1, norm=get_norm(norm, out_channels[i]), activation=relu)

            self.add_module("bottom_up_conv{}".format(i-1), bp_conv)
            self.add_module("output_conv{}".format(i), o_conv)
            bottom_up_convs.append(bp_conv)
            output_convs.append(o_conv)
        
        # weights init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                weight_init.c2_xavier_fill(m)

    def forward(self, x):
        assert len(x) == len(in_channels)
        results = [x[0]]
        output_feature = x[0]
        for i in range(1, len(x)):
            prev_features = bottom_up_convs[i-1](output_feature)
            fused = prev_features + x[i]
            output_feature = output_convs[i-1](fused)
            results.append(output_feature)
        return results

class ASFF(nn.Module):
    """
    Adaptive spatial feature fusion in https://arxiv.org/abs/1911.09516.
    """
    def __init__(self, in_channels=[256, 256, 256], out_channels=[256, 256, 256], norm="BN"):
        super(ASFF, self).__init__()
        assert len(in_channels) == 3 and len(out_channels) == 3

        h_in_channel = in_channels[0]
        m_in_channel = in_channels[1]
        l_in_channel = in_channels[2]

        h_out_channel = out_channels[0]
        m_out_channel = out_channels[1]
        l_out_channel = out_channels[2]

        self.in_channels = in_channels
        self.out_channels = out_channels

        relu = nn.ReLU(inplace=True)
        
        self.res_l_convs = [
            nn.Sequential(
                Conv2d(l_in_channel, h_in_channel, kernel_size=1, norm=get_norm(norm, h_in_channel), activation=relu),
                nn.UpsamplingNearest2d(scale_factor=4),
            ),
            nn.Sequential(
                Conv2d(l_in_channel, m_in_channel, kernel_size=1, norm=get_norm(norm, m_in_channel), activation=relu),
                nn.UpsamplingNearest2d(scale_factor=2),
            ),
            nn.Identity(),
        ]

        self.res_m_convs = [
            nn.Sequential(
                Conv2d(m_in_channel, h_in_channel, kernel_size=1, norm=get_norm(norm, h_in_channel), activation=relu),
                nn.UpsamplingNearest2d(scale_factor=2),
            ),
            nn.Identity(),
            Conv2d(m_in_channel, l_in_channel, kernel_size=3, stride=2, padding=1, norm=get_norm(norm, l_in_channel), activation=relu),
        ]

        self.res_h_convs = [
            nn.Identity(),
            Conv2d(h_in_channel, m_in_channel, kernel_size=3, stride=2, padding=1, norm=get_norm(norm, m_in_channel), activation=relu),
            nn.Sequential(
                nn.MaxPool2d(3, stride=2, padding=1),
                Conv2d(h_in_channel, l_in_channel, kernel_size=3, stride=2, padding=1, norm=get_norm(norm, l_in_channel), activation=relu),
            )
        ]

        self.output_convs = [
            Conv2d(h_in_channel, h_out_channel, kernel_size=3, padding=1, norm=get_norm(norm, h_out_channel), activation=relu),
            Conv2d(m_in_channel, m_out_channel, kernel_size=3, padding=1, norm=get_norm(norm, m_out_channel), activation=relu),
            Conv2d(l_in_channel, l_out_channel, kernel_size=3, padding=1, norm=get_norm(norm, l_out_channel), activation=relu),
        ]

        compress_c = 16

        self.weights_h_convs = [
            Conv2d(h_in_channel, compress_c, kernel_size=1, norm=get_norm(norm, compress_c), activation=relu),
            Conv2d(m_in_channel, compress_c, kernel_size=1, norm=get_norm(norm, compress_c), activation=relu),
            Conv2d(l_in_channel, compress_c, kernel_size=1, norm=get_norm(norm, compress_c), activation=relu),
        ]
        self.weights_m_convs = [
            Conv2d(h_in_channel, compress_c, kernel_size=1, norm=get_norm(norm, compress_c), activation=relu),
            Conv2d(m_in_channel, compress_c, kernel_size=1, norm=get_norm(norm, compress_c), activation=relu),
            Conv2d(l_in_channel, compress_c, kernel_size=1, norm=get_norm(norm, compress_c), activation=relu),
        ]

        self.weights_l_convs = [
            Conv2d(h_in_channel, compress_c, kernel_size=1, norm=get_norm(norm, compress_c), activation=relu),
            Conv2d(m_in_channel, compress_c, kernel_size=1, norm=get_norm(norm, compress_c), activation=relu),
            Conv2d(l_in_channel, compress_c, kernel_size=1, norm=get_norm(norm, compress_c), activation=relu),
        ]

        self.weights_compress = [
            Conv2d(compress_c*3, 3, kernel_size=1, stride=1, padding=0),
            Conv2d(compress_c*3, 3, kernel_size=1, stride=1, padding=0),
            Conv2d(compress_c*3, 3, kernel_size=1, stride=1, padding=0),
        ]

        self.add_module("h2h_resize", self.res_h_convs[0])
        self.add_module("h2m_resize", self.res_h_convs[1])
        self.add_module("h2l_resize", self.res_h_convs[2])
        self.add_module("m2h_resize", self.res_m_convs[0])
        self.add_module("m2m_resize", self.res_m_convs[1])
        self.add_module("m2l_resize", self.res_m_convs[2])
        self.add_module("l2h_resize", self.res_l_convs[0])
        self.add_module("l2m_resize", self.res_l_convs[1])
        self.add_module("l2l_resize", self.res_l_convs[2])

        self.add_module("h_output", self.output_convs[0])
        self.add_module("m_output", self.output_convs[1])
        self.add_module("l_output", self.output_convs[2])

        self.add_module("h2h_weights", self.weights_h_convs[0])
        self.add_module("h2m_weights", self.weights_h_convs[1])
        self.add_module("h2l_weights", self.weights_h_convs[2])
        self.add_module("m2h_weights", self.weights_m_convs[0])
        self.add_module("m2m_weights", self.weights_m_convs[1])
        self.add_module("m2l_weights", self.weights_m_convs[2])
        self.add_module("l2h_weights", self.weights_l_convs[0])
        self.add_module("l2m_weights", self.weights_l_convs[1])
        self.add_module("l2l_weights", self.weights_l_convs[2])

        self.add_module("h_weights", self.weights_compress[0])
        self.add_module("m_weights", self.weights_compress[1])
        self.add_module("l_weights", self.weights_compress[2])

        # weights init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                weight_init.c2_xavier_fill(m)
        
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
            x_out = self.output_convs[i](fused)
            results.append(x_out)

        return results

class BiFPNBlock(nn.Module):
    def __init__(self, num_channels, conv_channels=None, first_time=False, epsilon=1e-4, attention=True, norm="BN"):
        """
        Args:
            num_channels (int):
            conv_channels (list(int)): the 
            first_time: whether the input comes directly from the backbone,
                        if True, downchannel it first, and downsample s5 to generate s6 then s7
            epsilon: epsilon of fast weighted attention sum of BiFPN, not the BN's epsilon
        """
        super(BiFPNBlock, self).__init__()

        self.epsilon = epsilon
        # Conv layers
        
        self.conv6_up = Conv2d(num_channels, num_channels, 3, stride=1, padding=1, norm=get_norm(norm, num_channels))
        self.conv5_up = Conv2d(num_channels, num_channels, 3, stride=1, padding=1, norm=get_norm(norm, num_channels))
        self.conv4_up = Conv2d(num_channels, num_channels, 3, stride=1, padding=1, norm=get_norm(norm, num_channels))
        self.conv3_up = Conv2d(num_channels, num_channels, 3, stride=1, padding=1, norm=get_norm(norm, num_channels))
        
        self.conv4_down = Conv2d(num_channels, num_channels, 3, stride=1, padding=1, norm=get_norm(norm, num_channels))
        self.conv5_down = Conv2d(num_channels, num_channels, 3, stride=1, padding=1, norm=get_norm(norm, num_channels))
        self.conv6_down = Conv2d(num_channels, num_channels, 3, stride=1, padding=1, norm=get_norm(norm, num_channels))
        self.conv7_down = Conv2d(num_channels, num_channels, 3, stride=1, padding=1, norm=get_norm(norm, num_channels))

        # Feature scaling layers
        self.p6_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.p4_downsample = nn.MaxPool2d(3, stride=2, padding=1)
        self.p5_downsample = nn.MaxPool2d(3, stride=2, padding=1)
        self.p6_downsample = nn.MaxPool2d(3, stride=2, padding=1)
        self.p7_downsample = nn.MaxPool2d(3, stride=2, padding=1)

        self.activation = nn.ReLU()

        self.first_time = first_time
        if self.first_time:
            self.p5_down_channel = nn.Sequential(
                Conv2d(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p4_down_channel = nn.Sequential(
                Conv2d(conv_channels[1], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p3_down_channel = nn.Sequential(
                Conv2d(conv_channels[0], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )

            self.p5_to_p6 = nn.Sequential(
                Conv2d(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
                nn.MaxPool2d(3, stride=2, padding=1)
            )
            self.p6_to_p7 = nn.Sequential(
                nn.MaxPool2d(3, stride=2, padding=1)
            )

            self.p4_down_channel_2 = nn.Sequential(
                Conv2d(conv_channels[1], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p5_down_channel_2 = nn.Sequential(
                Conv2d(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )

        # Weight
        self.p6_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p6_w1_relu = nn.ReLU()
        self.p5_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_w1_relu = nn.ReLU()
        self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1_relu = nn.ReLU()

        self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p5_w2_relu = nn.ReLU()
        self.p6_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p6_w2_relu = nn.ReLU()
        self.p7_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p7_w2_relu = nn.ReLU()

        self.attention = attention

        # weights init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                weight_init.c2_xavier_fill(m)
        
    def forward(self, inputs):
        """
        illustration of a minimal bifpn unit
            P7_0 -------------------------> P7_2 -------->
               |-------------|                ↑
                             ↓                |
            P6_0 ---------> P6_1 ---------> P6_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P5_0 ---------> P5_1 ---------> P5_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P4_0 ---------> P4_1 ---------> P4_2 -------->
               |-------------|--------------↑ ↑
                             |--------------↓ |
            P3_0 -------------------------> P3_2 -------->
        """

        # downsample channels using same-padding conv2d to target phase's if not the same
        # judge: same phase as target,
        # if same, pass;
        # elif earlier phase, downsample to target phase's by pooling
        # elif later phase, upsample to target phase's by nearest interpolation

        if self.attention:
            p3_out, p4_out, p5_out, p6_out, p7_out = self._forward_fast_attention(inputs)
        else:
            p3_out, p4_out, p5_out, p6_out, p7_out = self._forward(inputs)

        return [p3_out, p4_out, p5_out, p6_out, p7_out]

    def _forward_fast_attention(self, inputs):
        if self.first_time:
            p3, p4, p5 = inputs

            p6_in = self.p5_to_p6(p5)
            p7_in = self.p6_to_p7(p6_in)

            p3_in = self.p3_down_channel(p3)
            p4_in = self.p4_down_channel(p4)
            p5_in = self.p5_down_channel(p5)

        else:
            # P3_0, P4_0, P5_0, P6_0 and P7_0
            p3_in, p4_in, p5_in, p6_in, p7_in = inputs

        # P7_0 to P7_2

        # Weights for P6_0 and P7_0 to P6_1
        p6_w1 = self.p6_w1_relu(self.p6_w1)
        weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
        # Connections for P6_0 and P7_0 to P6_1 respectively
        p6_up = self.conv6_up(self.activation(weight[0] * p6_in + weight[1] * self.p6_upsample(p7_in)))

        # Weights for P5_0 and P6_0 to P5_1
        p5_w1 = self.p5_w1_relu(self.p5_w1)
        weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
        # Connections for P5_0 and P6_0 to P5_1 respectively
        p5_up = self.conv5_up(self.activation(weight[0] * p5_in + weight[1] * self.p5_upsample(p6_up)))

        # Weights for P4_0 and P5_0 to P4_1
        p4_w1 = self.p4_w1_relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        # Connections for P4_0 and P5_0 to P4_1 respectively
        p4_up = self.conv4_up(self.activation(weight[0] * p4_in + weight[1] * self.p4_upsample(p5_up)))

        # Weights for P3_0 and P4_1 to P3_2
        p3_w1 = self.p3_w1_relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        # Connections for P3_0 and P4_1 to P3_2 respectively
        p3_out = self.conv3_up(self.activation(weight[0] * p3_in + weight[1] * self.p3_upsample(p4_up)))

        if self.first_time:
            p4_in = self.p4_down_channel_2(p4)
            p5_in = self.p5_down_channel_2(p5)

        # Weights for P4_0, P4_1 and P3_2 to P4_2
        p4_w2 = self.p4_w2_relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p4_out = self.conv4_down(
            self.activation(weight[0] * p4_in + weight[1] * p4_up + weight[2] * self.p4_downsample(p3_out)))

        # Weights for P5_0, P5_1 and P4_2 to P5_2
        p5_w2 = self.p5_w2_relu(self.p5_w2)
        weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
        p5_out = self.conv5_down(
            self.activation(weight[0] * p5_in + weight[1] * p5_up + weight[2] * self.p5_downsample(p4_out)))

        # Weights for P6_0, P6_1 and P5_2 to P6_2
        p6_w2 = self.p6_w2_relu(self.p6_w2)
        weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
        # Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
        p6_out = self.conv6_down(
            self.activation(weight[0] * p6_in + weight[1] * p6_up + weight[2] * self.p6_downsample(p5_out)))

        # Weights for P7_0 and P6_2 to P7_2
        p7_w2 = self.p7_w2_relu(self.p7_w2)
        weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
        # Connections for P7_0 and P6_2 to P7_2
        p7_out = self.conv7_down(self.activation(weight[0] * p7_in + weight[1] * self.p7_downsample(p6_out)))

        return [p3_out, p4_out, p5_out, p6_out, p7_out]

    def _forward(self, inputs):
        if self.first_time:
            p3, p4, p5 = inputs

            p6_in = self.p5_to_p6(p5)
            p7_in = self.p6_to_p7(p6_in)

            p3_in = self.p3_down_channel(p3)
            p4_in = self.p4_down_channel(p4)
            p5_in = self.p5_down_channel(p5)

        else:
            # P3_0, P4_0, P5_0, P6_0 and P7_0
            p3_in, p4_in, p5_in, p6_in, p7_in = inputs

        # P7_0 to P7_2

        # Connections for P6_0 and P7_0 to P6_1 respectively
        p6_up = self.conv6_up(self.activation(p6_in + self.p6_upsample(p7_in)))

        # Connections for P5_0 and P6_0 to P5_1 respectively
        p5_up = self.conv5_up(self.activation(p5_in + self.p5_upsample(p6_up)))

        # Connections for P4_0 and P5_0 to P4_1 respectively
        p4_up = self.conv4_up(self.activation(p4_in + self.p4_upsample(p5_up)))

        # Connections for P3_0 and P4_1 to P3_2 respectively
        p3_out = self.conv3_up(self.activation(p3_in + self.p3_upsample(p4_up)))

        if self.first_time:
            p4_in = self.p4_down_channel_2(p4)
            p5_in = self.p5_down_channel_2(p5)

        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p4_out = self.conv4_down(
            self.activation(p4_in + p4_up + self.p4_downsample(p3_out)))

        # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
        p5_out = self.conv5_down(
            self.activation(p5_in + p5_up + self.p5_downsample(p4_out)))

        # Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
        p6_out = self.conv6_down(
            self.activation(p6_in + p6_up + self.p6_downsample(p5_out)))

        # Connections for P7_0 and P6_2 to P7_2
        p7_out = self.conv7_down(self.activation(p7_in + self.p7_downsample(p6_out)))

        return [p3_out, p4_out, p5_out, p6_out, p7_out]


class BiFPG(nn.Module):
    def __init__(self, in_features, in_channels, out_channels, attention=True, norm="BN", asff=False):
        super().__init__()
        self.in_features = in_features
        self.asff_enabled = len(in_features) >= 3 and asff
        assert self.asff_enabled
        if self.asff_enabled:
            self.asff = ASFF(in_channels=[out_channels]*3, out_channels=[out_channels]*3, norm=norm)

        self.bifpn1 = BiFPNBlock(out_channels, in_channels, first_time=True, epsilon=1e-4, attention=attention, norm=norm)
        self.bifpn2 = BiFPNBlock(out_channels, first_time=False, epsilon=1e-4, attention=attention, norm=norm)
        self.bifpn3 = BiFPNBlock(out_channels, first_time=False, epsilon=1e-4, attention=attention, norm=norm)
    
    def forward(self, x):
        x = [x[f] for f in self.in_features]
        x = self.bifpn1(x)
        x = self.bifpn2(x)
        x = self.bifpn3(x)
        if self.asff_enabled:
            x[:3:] = self.asff(x[0], x[1], x[2])
        return x


class Yolov3FPG(nn.Module):
    def __init__(self, in_channels, out_channels, in_strides, in_features=["s3", "s4", "s5"], out_features=["p3", "p4", "p5"], top_block=None, norm="BN", fuse_type="sum", naive=False, panetff=False, asff=False):
        """
        Args:
            in_channels (list[int]): number of channels in the input feature maps.
            out_channels (list[int]): number of channels in the output feature maps.
            in_strides (list[int]): number of strides in the input feature maps.
            in_features (list[str]): names of the input feature maps coming
                from the backbone to which FPN is attached. For example, if the
                backbone produces ["res2", "res3", "res4"], any *contiguous* sublist
                of these may be used; order must be from high to low resolution.
            out_features (list[str]): names of the output feature maps 
                corresponding to the in_features. For example, in_features is ["s3", "s4", "s5"], the corresponding output features is ["p3", "p4", "p5"]. Extra output features maybe produced by top block.
            top_block (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list. The top_block
                further downsamples the feature map. It must have an attribute
                "num_levels", meaning the number of extra FPN levels added by
                this block, and "in_feature", which is a string representing
                its input feature (e.g., p5).
            norm (str): the normalization to use.
            fuse_type (str): types for fusing the top down features and the lateral
                ones. It can be "sum" (default), which sums up element-wise; or "avg",
                which takes the element-wise mean of the two.
            naive (bool): a boolean value indicates whether 
                top feature is used to generate output feature in each stage.
            panetff (bool): a boolean value indicates whether 
                using feature fusion proposed in PANet(https://arxiv.org/abs/1803.01534).
            asff (bool): a boolean value indicates whether using 
                adaptive spatial feature fusion in https://arxiv.org/abs/1911.09516.
        """
        super().__init__()

        assert len(in_features) == len(in_channels) and len(in_features) == len(in_strides) and len(in_features) > 0, "Info of input features should be valid and consistent."
        assert len(out_features) == len(out_channels) and len(out_channels) > 0, "Info of output features should be valid and consistent."
        if top_block is None:
            assert len(out_features) == len(in_features), "Output feature and input feature should be consistent when top block is none."
        
        norm="BN"
        # whether use asff module
        self.asff_enabled = len(in_features) >= 3 and len(out_features) >= 3 and asff
        self.panetff_enabled = panetff
        if self.asff_enabled:
            self.asff = ASFF(in_channels=out_channels[:3:], out_channels=out_channels[:3:], norm=norm)
        if self.panetff_enabled:
            self.pannetff = PANetFF(in_channels=out_channels, out_channels=out_channels, norm=norm)
        top_convs = []
        lateral_convs = []
        output_convs = []
        

        for idx, (in_channel, out_channel) in enumerate(zip(in_channels, out_channels)):
            top_out_channel = in_channel // 2
            lateral_out_channel = in_channel // 2
            up_sampling = (idx < len(in_features) - 1) and not naive
            first_in_channel = in_channel
            top_conv = None
            if up_sampling:
                top_conv = nn.Sequential(
                    Conv2dBNLeakyReLU(in_channel, top_out_channel, kernel_size=1, stride=1, padding=0, norm=norm),
                    nn.UpsamplingNearest2d(scale_factor=2)
                )
                first_in_channel = in_channel + top_out_channel
            lateral_conv = nn.Sequential(
                Conv2dBNLeakyReLU(first_in_channel, lateral_out_channel,
                                kernel_size=1, stride=1, padding=0, norm=norm),
                Conv2dBNLeakyReLU(lateral_out_channel, lateral_out_channel * 2,
                                kernel_size=3, stride=1, padding=1, norm=norm),
                Conv2dBNLeakyReLU(lateral_out_channel * 2, lateral_out_channel,
                                kernel_size=1, stride=1, padding=0, norm=norm),
                Conv2dBNLeakyReLU(lateral_out_channel, lateral_out_channel * 2,
                                kernel_size=3, stride=1, padding=1, norm=norm),
                Conv2dBNLeakyReLU(lateral_out_channel * 2, lateral_out_channel,
                                kernel_size=1, stride=1, padding=0, norm=norm),
            )
            output_conv = Conv2dBNLeakyReLU(lateral_out_channel, out_channel,
                                            kernel_size=3, stride=1, padding=1, norm=norm)
            stage = int(math.log2(in_strides[idx]))
            if up_sampling:
                self.add_module("fpn_top{}".format(stage), top_conv)
                top_convs.append(top_conv)
            self.add_module("fpn_lateral{}".format(stage), lateral_conv)
            self.add_module("fpn_output{}".format(stage), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]
        if not naive:
            self.top_convs = top_convs[::-1]
        
        self.naive = naive
        self.top_block = top_block
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, bottom_up_features):
        x = [bottom_up_features[f] for f in self.in_features[::-1]]
        results = []
        prev_features = self.lateral_convs[0](x[0])
        results.append(self.output_convs[0](prev_features))
        for idx, (features, lateral_conv, output_conv) in enumerate(zip(
            x[1:], self.lateral_convs[1:], self.output_convs[1:])
        ):
            if self.naive:
                x = features
            else:
                top_down_features = self.top_convs[idx](prev_features)
                x = torch.cat((top_down_features, features), dim=1)
            lateral_features = lateral_conv(x)
            prev_features = lateral_features

            results.insert(0, output_conv(lateral_features))
        if self.top_block is not None:
            top_block_in_feature = bottom_up_features.get(self.top_block.in_feature, None)
            if top_block_in_feature is None:
                top_block_in_feature = results[self.out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))
        if self.panetff_enabled:
            results = self.panetff(results)
        if self.asff_enabled:
            results[:3:] = self.asff(results[0], results[1], results[2])
        assert len(self.out_features) == len(results)
        return results

class RetinaFPG(nn.Module):
    def __init__(self, in_channels, out_channels, in_strides, in_features=["s3", "s4", "s5"], out_features=["p3", "p4", "p5"], top_block=None, norm="BN", fuse_type="sum", naive=False, panetff=False, asff=False):
        super().__init__()

        assert len(in_features) == len(in_channels) and len(in_features) == len(in_strides) and len(in_features) > 0, "Info of input features should be valid and consistent."
        assert len(out_features) == len(out_channels) and len(out_channels) > 0, "Info of output features should be valid and consistent."
        if top_block is None:
            assert len(out_features) == len(in_features), "Output feature and input feature should be consistent when top block is none."
        # whether use asff module
        self.asff_enabled = len(in_features) >= 3 and len(out_features) >= 3 and asff
        if self.asff_enabled:
            self.asff = ASFF(in_channels=out_channels[:3:], out_channels=out_channels[:3:], norm="BN")
        self.panetff_enabled = panetff
        if self.panetff_enabled:
            self.pannetff = PANetFF(in_channels=out_channels, out_channels=out_channels, norm=norm)
        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channel in enumerate(in_channels):
            out_channel = out_channels[idx]
            lateral_norm = get_norm(norm, out_channel)
            output_norm = get_norm(norm, out_channel)

            lateral_conv = Conv2d(
                in_channel, out_channel, kernel_size=1, bias=use_bias, norm=lateral_norm
            )
            output_conv = Conv2d(
                out_channel,
                out_channel,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm,
            )
            weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)
            stage = int(math.log2(in_strides[idx]))
            self.add_module("fpn_lateral{}".format(stage), lateral_conv)
            self.add_module("fpn_output{}".format(stage), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]
        self.top_block = top_block
        self.in_features = in_features
        self.out_features = out_features
        self.naive = naive

        assert fuse_type in {"avg", "sum"}
        self._fuse_type = fuse_type

    def forward(self, bottom_up_features):
        x = [bottom_up_features[f] for f in self.in_features[::-1]]
        results = []
        prev_features = self.lateral_convs[0](x[0])
        results.append(self.output_convs[0](prev_features))
        for features, lateral_conv, output_conv in zip(
            x[1:], self.lateral_convs[1:], self.output_convs[1:]
        ):
            if self.naive:
                prev_features = lateral_conv(features)
            else:
                top_down_features = F.interpolate(prev_features, scale_factor=2, mode="nearest")
                lateral_features = lateral_conv(features)
                prev_features = lateral_features + top_down_features
                if self._fuse_type == "avg":
                    prev_features /= 2
            results.insert(0, output_conv(prev_features))
        if self.top_block is not None:
            top_block_in_feature = bottom_up_features.get(self.top_block.in_feature, None)
            if top_block_in_feature is None:
                top_block_in_feature = results[self.out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))
        if self.panetff_enabled:
            results = self.panetff(results)
        if self.asff_enabled:
            results[:3:] = self.asff(results[0], results[1], results[2])
        assert len(self.out_features) == len(results)
        return results