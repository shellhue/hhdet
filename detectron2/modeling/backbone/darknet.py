import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init
import logging 

from detectron2.layers import (
    Conv2d,
    FrozenBatchNorm2d,
    get_norm,
    CNNBlockBase,
)

from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.backbone import Backbone

__all__ = [
    "Conv2dBNLeakyReLU",
    "DarknetBottleneckBlock",
    "DarknetBasicStem",
    "build_darknet53_backbone",
]

class Conv2dBNLeakyReLU(CNNBlockBase):
    def __init__(self,
                 in_channels,
                 out_channels,
                 *,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 norm="BN"):
        super(Conv2dBNLeakyReLU, self).__init__(
            in_channels, out_channels, stride)
        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn1 = get_norm(norm, out_channels)

        # parameter init
        weight_init.c2_xavier_fill(self.conv1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu_(x, negative_slope=0.1)
        return x


class DarknetBottleneckBlock(CNNBlockBase):
    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        bottleneck_channels,
        norm="BN",
    ):
        """
        Args:
            norm (str or callable): a callable that takes the number of
                channels and return a `nn.Module`, or a pre-defined string
                (one of {"FrozenBN", "BN", "GN"}).
        """
        super().__init__(in_channels, out_channels, 1)

        self.conv1 = Conv2dBNLeakyReLU(
            in_channels, bottleneck_channels, kernel_size=1, stride=1, padding=0, norm=norm)

        self.conv2 = Conv2dBNLeakyReLU(
            bottleneck_channels, out_channels, kernel_size=3, stride=1, padding=1, norm=norm)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out + x

def darknet_make_stage(block_class,
                       num_blocks,
                       in_channels,
                       out_channels,
                       *,
                       bottleneck_channels,
                       norm="BN",):
    """
    Create a resnet stage by creating many blocks.
    Args:
        block_class (class): a subclass of ResNetBlockBase
        num_blocks (int):
        kwargs: other arguments passed to the block constructor.

    Returns:
        list[nn.Module]: a list of block module.
    """
    blocks = [
        Conv2dBNLeakyReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            norm=norm)
    ]
    for _ in range(num_blocks):
        blocks.append(block_class(
            in_channels=out_channels,
            out_channels=out_channels,
            bottleneck_channels=bottleneck_channels,
            norm=norm))
    return blocks


class DarknetBasicStem(nn.Module):
    def __init__(self, in_channels=3, out_channels=32, norm="BN"):
        """
        Args:
            norm (str or callable): a callable that takes the number of
                channels and return a `nn.Module`, or a pre-defined string
                (one of {"FrozenBN", "BN", "GN"}).
        """
        super().__init__()
        self.conv1 = Conv2dBNLeakyReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            norm=norm
        )

    def forward(self, x):
        return self.conv1(x)

    @property
    def out_channels(self):
        return self.conv1.out_channels

    @property
    def stride(self):
        return 1


class Darknet(Backbone):
    def __init__(self, stem, stages, num_classes=1000, out_features=["linear"]):
        """
        Args:
            stem (nn.Module): a stem module
            stages (list[list[ResNetBlock]]): several (typically 4) stages,
                each contains multiple :class:`ResNetBlockBase`.
            num_classes (int): if 0, will not perform classification.
            out_features (list[str]): The features that will be outputed. Each value should be in ["s1", "s2", "s3", "s4", "s5", "avg", "linear"].
        """
        super(Darknet, self).__init__()
        if "linear" in out_features:
            assert num_classes > 0, "linear output needs num_classes bigger than zero."
        for f in out_features:
            assert f in ["s1", "s2", "s3", "s4", "s5", "avg", "linear"], "output feature: {} is not valid.".format(f)
        self._out_features = out_features
        self._out_feature_strides = {}
        self._out_feature_channels = {}

        self.stem = stem
        self.num_classes = num_classes

        current_stride = self.stem.stride

        self.stages_and_names = []
        for i, blocks in enumerate(stages):
            for block in blocks:
                assert isinstance(block, CNNBlockBase) or isinstance(
                    block, Conv2dBNLeakyReLU), block
                curr_channels = block.out_channels
            stage = nn.Sequential(*blocks)
            name = "s" + str(i + 1)
            self.add_module(name, stage)
            self.stages_and_names.append((stage, name))
            self._out_feature_strides[name] = current_stride = int(
                current_stride * np.prod([k.stride for k in blocks])
            )
            self._out_feature_channels[name] = blocks[-1].out_channels

        if "linear" in self._out_features or "avg" in self._out_features:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        if "linear" in self._out_features:
            self.linear = nn.Linear(curr_channels, num_classes)
            nn.init.normal_(self.linear.weight, std=0.01)

            # self.linear = Conv2d(
            #     curr_channels,
            #     num_classes,
            #     kernel_size=1,
            #     stride=1,
            #     bias=True
            # )
            # Sec 5.1 in "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour":
            # "The 1000-way fully-connected layer is initialized by
            # drawing weights from a zero-mean Gaussian with standard deviation of 0.01."
            
            # weight_init.c2_msra_fill(self.linear)

    def forward(self, x, targets=None):
        outputs = {}
        x = self.stem(x)
        if "stem" in self._out_features:
            outputs["stem"] = x
        for stage, name in self.stages_and_names:
            x = stage(x)
            outputs[name] = x
        
        if "linear" in self._out_features or "avg" in self._out_features:
            x = self.avgpool(x)
            x = nn.Flatten()(x)
            outputs["avg"] = x

        if "linear" in self._out_features:
            x = self.linear(x)
            outputs["linear"] = x
        
        return {f: outputs[f] for f in self._out_features}

    def get_conv_bn_modules(self, include_linear=False):
        """
        for weight convert from original yolo weights file
        """
        modules = []
        count = 0
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                modules.append(module)
            if isinstance(module, nn.BatchNorm2d):
                modules.append(module)
            if include_linear and isinstance(module, nn.Linear):
                modules.append(module)
        return modules
    
    def freeze(self, freeze_at=0):
        if freeze_at <= 0:
            return
        if freeze_at >= 1:
            for p in self.stem.parameters():
                p.requires_grad = False
            self.stem = FrozenBatchNorm2d.convert_frozen_batchnorm(self.stem)
        
        for stage_idx, (stage, name) in enumerate(self.stages_and_names):
            if freeze_at >= (stage_idx+1):
                for block in stage:
                    block.freeze()

def load_darknet_weights(weights, modules):
    logger = logging.getLogger(__name__)
    logger.info("Starting to convert darknet53 pretrained weights from file: {}".format(weights))
    with open(weights, 'rb') as f:
        # (int32) version info: major, minor, revision
        version = np.fromfile(f, dtype=np.int32, count=3)
        # (int64) number of images seen during training
        seen = np.fromfile(f, dtype=np.int64, count=1)
        # the rest are weights
        weights = np.fromfile(f, dtype=np.float32)
        logger.info("Weigths version: {} . Number of images seen during training: {}".format(version, seen))
        logger.info("Param count in pretrained weights file: {}".format(weights.shape))

    ptr = 0
    paired_modules = []
    param_count = 0
    for i, module in enumerate(modules):
        if isinstance(module, nn.Conv2d):
            if not module.bias is None:
                paired_modules.append([module])
                param_count += module.weight.numel()
                param_count += module.bias.numel()
            else:
                paired_modules.append([module, modules[i+1]])
                param_count += module.weight.numel()
                param_count += modules[i+1].bias.numel() * 4
        elif isinstance(module, nn.Linear):
            paired_modules.append([module])
            param_count += module.weight.numel()
            param_count += module.bias.numel()
    logger.info("Param count in darknet53: {}".format(param_count))
    for ms in paired_modules:
        # When loading weights from darknet53.conv.74, some parameters in model are not missing.
        if weights.shape[0] <= ptr:
            break
        if isinstance(ms[0], nn.Conv2d):
            conv = ms[0]
            conv_bn_modules = ms
            bn = conv_bn_modules[1] if len(conv_bn_modules) == 2 else None
            out_channel, in_channel, kernel_h, kernel_w = conv.weight.size()
            if bn:
                assert bn.bias.size()[0] == out_channel, "conv and bn is not paired"
                # Bias
                bn_b = torch.from_numpy(weights[ptr:ptr + out_channel]).view_as(bn.bias)
                bn.bias.data.copy_(bn_b)
                ptr += out_channel
                # Weight
                bn_w = torch.from_numpy(weights[ptr:ptr + out_channel]).view_as(bn.weight)
                bn.weight.data.copy_(bn_w)
                ptr += out_channel
                # Running Mean
                bn_rm = torch.from_numpy(weights[ptr:ptr + out_channel]).view_as(bn.running_mean)
                bn.running_mean.data.copy_(bn_rm)
                ptr += out_channel
                # Running Var
                bn_rv = torch.from_numpy(weights[ptr:ptr + out_channel]).view_as(bn.running_var)
                bn.running_var.data.copy_(bn_rv)
                ptr += out_channel
            else:
                # Load conv. bias
                conv_b = torch.from_numpy(weights[ptr:ptr + out_channel]).view_as(conv.bias)
                conv.bias.data.copy_(conv_b)
                ptr += out_channel
            # Load conv. weights
            num_w = conv.weight.numel()
            conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv.weight)
            conv.weight.data.copy_(conv_w)
            ptr += num_w
        elif isinstance(ms[0], nn.Linear):
            linear = ms[0]
            
            num_b = linear.bias.numel()
            linear_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(linear.bias)
            linear.bias.data.copy_(linear_b)
            ptr += num_b

            num_w = linear.weight.numel()
            linear_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(linear.weight)
            linear.weight.data.copy_(linear_w)
            ptr += num_w

    logger.info("Parsed parmeter count: {}".format(ptr))
    logger.info("Darknet53 pretrained weights converted succeed.") 

@BACKBONE_REGISTRY.register()
def build_darknet53_backbone(cfg, input_shape):
    """
    Create a ResNet instance from config.

    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    # need registration of new blocks/stems?
    norm = cfg.MODEL.DARKNET53.NORM
    stem = DarknetBasicStem(
        in_channels=input_shape.channels,
        out_channels=cfg.MODEL.DARKNET53.STEM_OUT_CHANNELS,
        norm=norm,
    )

    out_features = cfg.MODEL.DARKNET53.OUT_FEATURES
    in_channels = cfg.MODEL.DARKNET53.STEM_OUT_CHANNELS

    num_blocks_per_stage = [1, 2, 8, 8, 4]

    stages = []

    # Avoid creating variables without gradients
    # It consumes extra memory and may cause allreduce to fail
    if "linear" in out_features:
        max_stage_idx = 5
    else:
        out_stage_idx = [{"s1": 1, "s2": 2, "s3": 3, "s4": 4, "s5": 5}[f]
                        for f in out_features]
        max_stage_idx = max(out_stage_idx)
    for idx, stage_idx in enumerate(range(1, max_stage_idx + 1)):
        stage_kargs = {
            "block_class": DarknetBottleneckBlock,
            "num_blocks": num_blocks_per_stage[idx],
            "in_channels": in_channels,
            "bottleneck_channels": in_channels,
            "out_channels": in_channels * 2,
            "norm": norm,
        }
        blocks = darknet_make_stage(**stage_kargs)
        in_channels *= 2

        stages.append(blocks)

    model = Darknet(stem, stages, out_features=out_features, num_classes=cfg.MODEL.DARKNET53.NUM_CLASSES)

    pretrained_weights = cfg.MODEL.DARKNET53.PRETRAINED_WEIGHTS
    if len(pretrained_weights) == 0:
        return model
    modules = model.get_conv_bn_modules(include_linear=("linear" in model._out_features))
    logger = logging.getLogger(__name__)
    if pretrained_weights.endswith(".weights"):
        load_darknet_weights(pretrained_weights, modules)
        logger.info("Backbone pretrained weights loaded.")
    elif pretrained_weights.endswith(".pth") or pretrained_weights.endswith(".pt"):
        # If map location is on cuda, unecessary extra cuda memory is needed. so map location is best on cpu.
        pretrained_state_dict = torch.load(pretrained_weights, map_location=torch.device('cpu'))
        if "model" in pretrained_state_dict:
            pretrained_state_dict = pretrained_state_dict["model"]
        n_pretrained_state_dict = {}
        for k, v in pretrained_state_dict.items():
            prefix = "backbone."
            if k.startswith("backbone."):
                n_pretrained_state_dict[k[len(prefix):]] = v
            else:
                n_pretrained_state_dict[k] = v
        # TODO: False strict makes some loading error silent, it is not good.
        model.load_state_dict(n_pretrained_state_dict, strict=False)
        logger.info("Backbone pretrained weights loaded from {}.".format(pretrained_weights))
    elif pretrained_weights.endswith(".conv.74"):
        load_darknet_weights(pretrained_weights, modules)
        logger.info("Backbone pretrained weights loaded.")
    # freeze model before stage
    model.freeze(cfg.MODEL.BACKBONE.FREEZE_AT)

    return model
