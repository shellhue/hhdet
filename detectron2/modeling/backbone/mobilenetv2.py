from torch import nn
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.backbone import Backbone
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

__all__ = ['MobileNetV2', 'mobilenet_v2']


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo. It ensures that all layers have a channel number that is divisible by divisor. It can be seen here: https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py

    Args:
        v (int): 
        divisor (int): 
        min_value (int): 

    Returns:
        int: 
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

def get_blocks(setting, input_channel, block, width_mult=1., round_nearest=8):
    features = []
    t, c, n, s = setting
    output_channel = _make_divisible(c * width_mult, round_nearest)
    for i in range(n):
        stride = s if i == 0 else 1
        features.append(block(input_channel, output_channel, stride, expand_ratio=t))
        input_channel = output_channel
    return features, output_channel


class MobileNetV2(Backbone):
    def __init__(self,
                 num_classes=1000,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 out_features=["linear"],
                 block=None):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number. Set to 1 to turn off rounding
            out_features (list[str]): The features that will be outputed. Each value should be in ["s1", "s2", "s3", "s4", "s5", "avg", "linear"].
            block: Module specifying inverted residual building block for mobilenet
        """
        super(MobileNetV2, self).__init__()
        self._out_features = out_features

        for f in self._out_features:
            assert f in ["s1", "s2", "s3", "s4", "s5", "avg", "linear"], "output feature: {} is not valid.".format(f)
        
        self._out_feature_channels = {}
        self._out_feature_strides = {
            "s1": 2,
            "s2": 4,
            "s3": 8,
            "s4": 16,
            "s5": 32,
        }

        if block is None:
            block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)

        # stage1
        input_channel = 32
        features = [ConvBNReLU(3, input_channel, stride=2)]
        fs, input_channel = get_blocks([1, 16, 1, 1], input_channel, block, width_mult=width_mult, round_nearest=round_nearest)
        features.extend(fs)
        self.stage1 = nn.Sequential(*features)
        self._out_feature_channels["s1"] = input_channel
        
        # stage2
        features = []
        fs, input_channel = get_blocks([6, 24, 2, 2], input_channel, block, width_mult=width_mult, round_nearest=round_nearest)
        features.extend(fs)
        self.stage2 = nn.Sequential(*features)
        self._out_feature_channels["s2"] = input_channel

        # stage3
        features = []
        fs, input_channel = get_blocks([6, 32, 3, 2], input_channel, block, width_mult=width_mult, round_nearest=round_nearest)
        features.extend(fs)
        self.stage3 = nn.Sequential(*features)
        self._out_feature_channels["s3"] = input_channel

        # stage4
        features = []
        fs, input_channel = get_blocks([6, 64, 4, 2], input_channel, block, width_mult=width_mult, round_nearest=round_nearest)
        features.extend(fs)
        fs, input_channel = get_blocks([6, 96, 3, 1], input_channel, block, width_mult=width_mult, round_nearest=round_nearest)
        features.extend(fs)
        self.stage4 = nn.Sequential(*features)
        self._out_feature_channels["s4"] = input_channel

        # stage5
        features = []
        fs, input_channel = get_blocks([6, 160, 3, 2], input_channel, block, width_mult=width_mult, round_nearest=round_nearest)
        features.extend(fs)
        fs, input_channel = get_blocks([6, 320, 1, 1], input_channel, block, width_mult=width_mult, round_nearest=round_nearest)
        features.extend(fs)
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        self.stage5 = nn.Sequential(*features)
        self._out_feature_channels["s5"] = self.last_channel

        if "linear" in self._out_features:
            # building classifier
            self.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(self.last_channel, num_classes),
            )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        output = {}
        x = self.stage1(x)
        output["stage1"] = x
        x = self.stage2(x)
        output["stage2"] = x
        x = self.stage3(x)
        output["stage3"] = x
        x = self.stage4(x)
        output["stage4"] = x
        x = self.stage5(x)
        output["stage5"] = x

        if "avg" in self._out_features:
            x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
            output["avg"] = x
        if "linear" in self._out_features:
            # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
            if "avg" not in self._out_features:
                x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
            x = self.classifier(x)
            output["linear"] = x

        return {f: output[f] for f in self._out_features}

    def forward(self, x):
        return self._forward_impl(x)

def convert_state_dict(previous_stage_dict):
    # stage1  features.0-1
    # stage2  features.2-3
    # stage3  features.4-6
    # stage4  features.7-13
    # stage5  features.14-18
    new_state_dict = {}
    for k, v in previous_stage_dict.items():
        if k.startswith("features."):
            components = k.split(".")
            order = int(components[1])
            if order <= 1:
                components[0] = "stage1"
            elif order <= 3:
                components[0] = "stage2"
                components[1] = str(order - 2)
            elif order <= 6:
                components[0] = "stage3"
                components[1] = str(order - 4)
            elif order <= 13:
                components[0] = "stage4"
                components[1] = str(order - 7)
            else:
                components[0] = "stage5"
                components[1] = str(order - 14)
            nk = ".".join(components)
            new_state_dict[nk] = v
        else:
            new_state_dict[k] = v

    return new_state_dict


def mobilenet_v2(pretrained=False, progress=True, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                              progress=progress)
        state_dict = convert_state_dict(state_dict)
        model.load_state_dict(state_dict)
    return model


@BACKBONE_REGISTRY.register()
def build_mobilenetv2_backbone(cfg, input_shape):
    model = mobilenet_v2(pretrained=True)

    return model