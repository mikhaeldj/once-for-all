import torch.nn as nn

from ofa.utils.layers import (
    set_layer_from_config,
    ConvLayer,
    IdentityLayer,
    LinearLayer,
)
from ofa.utils.layers import ResNetBottleneckBlock, ResidualBlock
from ofa.utils import make_divisible, MyNetwork, MyGlobalAvgPool2d

__all__ = ["ViT"]


class ViT(MyNetwork):
    def __init__(self, input_stem, blocks, classifier):
        super(ViT, self).__init__()

        self.input_stem = nn.ModuleList(input_stem)
        self.blocks = nn.ModuleList(blocks)
        self.classifier = classifier

    def forward(self, x):
        for layer in self.input_stem:
            x = layer(x)
        for block in self.blocks:
            x = block(x)
        x = self.classifier(x)
        return x

    @property
    def module_str(self):
        _str = ""
        for layer in self.input_stem:
            _str += layer.module_str + "\n"
        for layer in self.blocks[:self.active_depth]:
            _str += self.blocks.module_str + "\n"
        _str += self.classifier.module_str
        return _str

    @property
    def config(self):
        return {
            "name": ViT.__name__,
            "bn": self.get_bn_param(),
            "input_stem": [layer.config for layer in self.input_stem],
            "blocks": [block.config for block in self.blocks],
            "classifier": self.classifier.config,
        }

    @staticmethod
    def build_from_config(config):
        classifier = set_layer_from_config(config["classifier"])

        input_stem = []
        for layer_config in config["input_stem"]:
            input_stem.append(set_layer_from_config(layer_config))
        blocks = []
        for block_config in config["blocks"]:
            blocks.append(set_layer_from_config(block_config))

        net = ViT(input_stem, blocks, classifier)
        if "bn" in config:
            net.set_bn_param(**config["bn"])
        else:
            net.set_bn_param(momentum=0.1, eps=1e-5)

        return net

    def zero_last_gamma(self):
        for m in self.modules():
            if isinstance(m, ResNetBottleneckBlock) and isinstance(
                m.downsample, IdentityLayer
            ):
                m.conv3.bn.weight.data.zero_()

    @property
    def grouped_block_index(self):
        info_list = []
        block_index_list = []
        for i, block in enumerate(self.blocks):
            if (
                not isinstance(block.downsample, IdentityLayer)
                and len(block_index_list) > 0
            ):
                info_list.append(block_index_list)
                block_index_list = []
            block_index_list.append(i)
        if len(block_index_list) > 0:
            info_list.append(block_index_list)
        return info_list

    def load_state_dict(self, state_dict, **kwargs):
        super(ViT, self).load_state_dict(state_dict)
