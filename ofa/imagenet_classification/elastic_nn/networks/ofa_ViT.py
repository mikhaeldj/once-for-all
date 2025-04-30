import random

from ofa.imagenet_classification.elastic_nn.modules.dynamic_layers import (
    DinamicLinearMapper,
    DynamicCLSToken,
    DynamicPositionalEmbedding,
)
from ofa.imagenet_classification.elastic_nn.modules.dynamic_layers import (
    DynamicTransfromerBlock,
)
from ofa.utils.layers import SimpleLinearLayer
from ofa.imagenet_classification.networks import ViT
from ofa.utils import make_divisible, val2list, MyNetwork

__all__ = ["OFAViT"]


class OFAViT(ViT):
    def __init__(
        self,
        image_size=224,
        n_classes=1000,
        dim=16,
        dropout_rate=0,
        dim_heads=16,
        act_func="gelu",
    ):

        self.heads_list = [2, 4, 8, 16]
        self.depth_list = [2, 3, 4, 5, 6]
        self.width_mult_list = [2, 4, 8, 16]
        #self.patch_size_list =[2, 4, 8, 14, 16, 28, 56, 112] # with image_size = 224

        # max
        self.max_heads = max(self.heads_list)
        self.max_depth = max(self.depth_list)
        self.max_width_mult = max(self.width_mult_list)
        self.max_patch_size = max(self.patch_size_list)

        # build input stem
        input_stem = [
            DinamicLinearMapper(
                dim,
                image_size,
                self.max_patch_size,
            ),
            DynamicCLSToken(
                dim,
            ),
            DynamicPositionalEmbedding(
                dim,
                image_size,
                self.max_patch_size,
            ),
        ]

        # blocks
        blocks = []
        for _ in range(self.max_depth):
            transformer_block = DynamicTransfromerBlock(
                self,
                dim,
                self.max_heads,
                dim_heads,
                self.max_width_mult,
                dropout_rate,
                act_func,
            )
            blocks.append(transformer_block)

        # classifier
        classifier = SimpleLinearLayer(
            dim, n_classes, dropout_rate=dropout_rate
        )

        super(OFAViT, self).__init__(input_stem, blocks, classifier)

    @staticmethod
    def name():
        return "OFAViT"

    def forward(self, x, active_heads = None, active_width_mult = None, active_depth = None):
        if active_depth is None:
            active_depth = self.max_depth

        for layer in self.input_stem:
            x = layer(x)
        for block in self.blocks[:active_depth]:
            x = block(x, active_heads, active_width_mult)
        x = self.classifier(x)
        return x


    """ set, sample and get active sub-networks """
    '''
    def set_active_subnet(self, d=None, e=None, w=None, **kwargs):
        depth = val2list(d, len(ViT.BASE_DEPTH_LIST) + 1)
        expand_ratio = val2list(e, len(self.blocks))
        width_mult = val2list(w, len(ViT.BASE_DEPTH_LIST) + 2)

        for block, e in zip(self.blocks, expand_ratio):
            if e is not None:
                block.active_expand_ratio = e

        if width_mult[0] is not None:
            self.input_stem[1].conv.active_out_channel = self.input_stem[
                0
            ].active_out_channel = self.input_stem[0].out_channel_list[width_mult[0]]
        if width_mult[1] is not None:
            self.input_stem[2].active_out_channel = self.input_stem[2].out_channel_list[
                width_mult[1]
            ]

        if depth[0] is not None:
            self.input_stem_skipping = depth[0] != max(self.depth_list)
        for stage_id, (block_idx, d, w) in enumerate(
            zip(self.grouped_block_index, depth[1:], width_mult[2:])
        ):
            if d is not None:
                self.runtime_depth[stage_id] = max(self.depth_list) - d
            if w is not None:
                for idx in block_idx:
                    self.blocks[idx].active_out_channel = self.blocks[
                        idx
                    ].out_channel_list[w]
    '''
                    
    def sample_active_subnet(self):
        heads_candidates = self.heads_list
        width_mult_candidates = self.width_mult_list
        depth_candidates = self.depth_list

        # sample kernel size
        heads_setting = []
        if not isinstance(heads_candidates[0], list):
            heads_candidates = [heads_candidates for _ in range(len(self.blocks) - 1)]
        for k_set in heads_candidates:
            k = random.choice(k_set)
            heads_setting.append(k)

        # sample expand ratio
        width_mult_setting = []
        if not isinstance(width_mult_candidates[0], list):
            width_mult_candidates = [width_mult_candidates for _ in range(len(self.blocks) - 1)]
        for e_set in width_mult_candidates:
            e = random.choice(e_set)
            width_mult_setting.append(e)

        # sample depth
        depth_setting = []
        if not isinstance(depth_candidates[0], list):
            depth_candidates = [
                depth_candidates for _ in range(len(self.block_group_info))
            ]
        for d_set in depth_candidates:
            d = random.choice(d_set)
            depth_setting.append(d)

        self.set_active_subnet(heads_setting, width_mult_setting, depth_setting)

        return {
            "h": heads_setting,
            "wm": width_mult_setting,
            "d": depth_setting,
        }
'''
    def get_active_subnet(self, preserve_weight=True):
        input_stem = [self.input_stem[0].get_active_subnet(3, preserve_weight)]
        if self.input_stem_skipping <= 0:
            input_stem.append(
                ResidualBlock(
                    self.input_stem[1].conv.get_active_subnet(
                        self.input_stem[0].active_out_channel, preserve_weight
                    ),
                    IdentityLayer(
                        self.input_stem[0].active_out_channel,
                        self.input_stem[0].active_out_channel,
                    ),
                )
            )
        input_stem.append(
            self.input_stem[2].get_active_subnet(
                self.input_stem[0].active_out_channel, preserve_weight
            )
        )
        input_channel = self.input_stem[2].active_out_channel

        blocks = []
        for stage_id, block_idx in enumerate(self.grouped_block_index):
            depth_param = self.runtime_depth[stage_id]
            active_idx = block_idx[: len(block_idx) - depth_param]
            for idx in active_idx:
                blocks.append(
                    self.blocks[idx].get_active_subnet(input_channel, preserve_weight)
                )
                input_channel = self.blocks[idx].active_out_channel
        classifier = self.classifier.get_active_subnet(input_channel, preserve_weight)
        subnet = ViT(input_stem, blocks, classifier)

        subnet.set_bn_param(**self.get_bn_param())
        return subnet

    def get_active_net_config(self):
        input_stem_config = [self.input_stem[0].get_active_subnet_config(3)]
        if self.input_stem_skipping <= 0:
            input_stem_config.append(
                {
                    "name": ResidualBlock.__name__,
                    "conv": self.input_stem[1].conv.get_active_subnet_config(
                        self.input_stem[0].active_out_channel
                    ),
                    "shortcut": IdentityLayer(
                        self.input_stem[0].active_out_channel,
                        self.input_stem[0].active_out_channel,
                    ),
                }
            )
        input_stem_config.append(
            self.input_stem[2].get_active_subnet_config(
                self.input_stem[0].active_out_channel
            )
        )
        input_channel = self.input_stem[2].active_out_channel

        blocks_config = []
        for stage_id, block_idx in enumerate(self.grouped_block_index):
            depth_param = self.runtime_depth[stage_id]
            active_idx = block_idx[: len(block_idx) - depth_param]
            for idx in active_idx:
                blocks_config.append(
                    self.blocks[idx].get_active_subnet_config(input_channel)
                )
                input_channel = self.blocks[idx].active_out_channel
        classifier_config = self.classifier.get_active_subnet_config(input_channel)
        return {
            "name": ViT.__name__,
            "bn": self.get_bn_param(),
            "input_stem": input_stem_config,
            "blocks": blocks_config,
            "classifier": classifier_config,
        }
    '''

""" Width Related Methods """

'''
    def re_organize_middle_weights(self, expand_ratio_stage=0):
        for block in self.blocks:
            block.re_organize_middle_weights(expand_ratio_stage)
'''
