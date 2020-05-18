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
    ]

class Yolov3FPG(nn.Module):
    def __init__(self, in_channels, out_channels, in_strides, in_features=["s3", "s4", "s5"], out_features=["p3", "p4", "p5"], top_block=None, norm="BN", fuse_type="sum", naive=False):
        """
        Args:
            norm (str or callable): a callable that takes the number of
                channels and return a `nn.Module`, or a pre-defined string
                (one of {"FrozenBN", "BN", "GN"}).
        """
        super().__init__()

        assert len(in_features) == len(in_channels) and len(in_features) == len(in_strides) and len(in_features) > 0, "Info of input features should be valid and consistent."
        assert len(out_features) == len(out_channels) and len(out_channels) > 0, "Info of output features should be valid and consistent."
        if top_block is None:
            assert len(out_features) == len(in_features), "Output feature and input feature should be consistent when top block is none."
        
        top_convs = []
        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, (in_channel, out_channel) in enumerate(zip(in_channels, out_channels)):
            top_out_channel = in_channel / 2
            lateral_out_channel = in_channel / 2
            up_sampling = (idx < len(in_features) - 1) and not naive
            first_in_channel = in_channel
            top = None
            if up_sampling:
                top = nn.Sequential(
                    Conv2dBNLeakyReLU(in_channel, top_out_channel,
                                    kernel_size=1, stride=1, padding=0, norm=norm),
                    nn.UpsamplingNearest2d(scale_factor=2)
                )
                first_in_channel = in_channel + top_out_channel
            lateral = nn.Sequential(
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
            output = Conv2dBNLeakyReLU(lateral_out_channel, out_channel,
                                            kernel_size=3, stride=1, padding=1, norm=norm)
            stage = int(math.log2(in_strides[idx]))
            if up_sampling:
                self.add_module("fpn_top{}".format(stage), top)
                top_convs.append(self.top)
            self.add_module("fpn_lateral{}".format(stage), lateral)
            self.add_module("fpn_output{}".format(stage), output)

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
        assert len(self.out_features) == len(results)
        return results

class RetinaFPG(nn.Module):
    def __init__(self, in_channels, out_channels, in_strides, in_features=["s3", "s4", "s5"], out_features=["p3", "p4", "p5"], top_block=None, norm="BN", fuse_type="sum", naive=False):
        """
        Args:
            norm (str or callable): a callable that takes the number of
                channels and return a `nn.Module`, or a pre-defined string
                (one of {"FrozenBN", "BN", "GN"}).
        """
        super().__init__()

        assert len(in_features) == len(in_channels) and len(in_features) == len(in_strides) and len(in_features) > 0, "Info of input features should be valid and consistent."
        assert len(out_features) == len(out_channels) and len(out_channels) > 0, "Info of output features should be valid and consistent."
        if top_block is None:
            assert len(out_features) == len(in_features), "Output feature and input feature should be consistent when top block is none."
        
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
        assert len(self.out_features) == len(results)
        return results