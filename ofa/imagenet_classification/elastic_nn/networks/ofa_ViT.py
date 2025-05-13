import random

from ofa.imagenet_classification.elastic_nn.modules.dynamic_layers import (
    DynamicLinearMapper,
    DynamicCLSToken,
    DynamicPositionalEmbedding,
    DynamicTransformerBlock,
)
from ofa.utils.layers import SimpleLinearLayer
from ofa.imagenet_classification.networks import ViT

__all__ = ["OFAViT"]


class OFAViT(ViT):
    def __init__(
        self,
        n_classes=1000,
        dropout_rate=0,
        image_size=32,
        dim=128,
        dim_heads=24,
        act_func="gelu",
        heads_list=None,
        width_mult_ratio_list=None,
        depth_list=None,
    ):
        self.classes = n_classes
        self.image_size = image_size
        self.dim = dim
        self.dim_heads = dim_heads
        self.act_func = act_func

        if heads_list is None:
            heads_list = [2, 4, 8, 16]
        if width_mult_ratio_list is None:
            width_mult_ratio_list = [2, 4, 8]
        if depth_list is None:
            depth_list = [2, 3, 4, 5, 6]
        if image_size == 32:
            patch_size_list = [2, 4, 8, 16]
        elif image_size == 224:
            patch_size_list = [2, 4, 8, 14, 16, 28, 56, 112]


        self.heads_list = heads_list
        self.width_mult_list = width_mult_ratio_list
        self.depth_list = depth_list
        self.patch_size_list = [2, 4, 8, 16]

        # max
        self.max_heads = max(self.heads_list)
        self.max_depth = max(self.depth_list)
        self.max_width_mult = max(self.width_mult_list)
        self.max_patch_size = max(patch_size_list)

        self.active_patch_size = 4

        self.active_depth = self.max_depth

        # build input stem
        input_stem = [
            DynamicLinearMapper(
                dim,
                image_size,
                self.active_patch_size,
            ),
            DynamicCLSToken(
                dim,
            ),
            DynamicPositionalEmbedding(
                dim,
                image_size,
                self.active_patch_size,
            ),
        ]

        # blocks
        blocks = []
        for _ in range(self.max_depth):
            transformer_block = DynamicTransformerBlock(
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

    """ MyNetwork required methods """

    @staticmethod
    def name():
        return "OFAViT"

    def forward(self, x, active_heads = None, active_width_mult = None, active_depth = None):
        if active_depth is None:
            active_depth = self.active_depth

        for layer in self.input_stem:
            x = layer(x)
        for block in self.blocks[:active_depth]:
            x = block(x, active_heads, active_width_mult)
        x = x[:, 0, :]
        x = self.classifier(x)
        return x
    
    @property
    def module_str(self):
        _str = ""
        for layer in self.input_stem:
            if hasattr(layer, 'module_str'):
                _str += layer.module_str + "\n"
            else:
                _str += str(layer) + "\n"
        
        _str += str(self.active_depth) + "x"
        if hasattr(self.blocks, 'module_str'):
            _str += self.blocks.module_str + "\n"
        else:
            _str += str(self.blocks) + "\n"
        
        if hasattr(self.classifier, 'module_str'):
            _str += self.classifier.module_str
        else:
            _str += str(self.classifier)
        
        return _str
    
    # @property
    # def module_str(self):
    #     print("Input stem types:", [type(layer) for layer in self.input_stem])
    #     print("Blocks types:", [type(block) for block in self.blocks])
    #     print("Classifier type:", type(self.classifier))
    #     _str = ""
    #     for layer in self.input_stem:
    #         _str += layer.module_str + "\n"
    #     for layer in self.blocks[:self.active_depth]:
    #         _str += self.blocks.module_str + "\n"
    #     _str += self.classifier.module_str
    #     return _str

    @property
    def config(self):
        return {
            "name": OFAViT.__name__,
            "input_stem": [layer.config for layer in self.input_stem],
            "blocks": [block.config for block in self.blocks],
            "classifier": self.classifier.config,
        }
    
    @staticmethod
    def build_from_config(config):
        raise ValueError("do not support this function")
    

    """ set, sample and get active sub-networks """

    def set_active_subnet(self, d=None, wm=None, h=None, **kwargs):
        for block in self.blocks:
            block.attention.active_heads = h
            block.attention.to_out.active_in_feature = h * self.dim_heads
            block.feedforward.active_width_mult = wm

        self.active_depth = d


    def sample_active_subnet(self):
        heads_candidates = self.heads_list
        width_mult_candidates = self.width_mult_list
        depth_candidates = self.depth_list

        # Sample random values correctly
        h = random.choice(heads_candidates) 
        w = random.choice(width_mult_candidates)
        d = random.choice(depth_candidates)

        self.set_active_subnet(d=d, wm=w, h=h) 

        return {
            "d": d,
            "wm": w,
            "h": h,
        }
    
    # def load_state_dict(self, state_dict, **kwargs):
    #     super(OFAViT, self).load_state_dict(state_dict)

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
