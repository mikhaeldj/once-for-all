# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import argparse
import numpy as np
import os
import random

import torch

from ofa.imagenet_classification.elastic_nn.modules.dynamic_op import (
    DynamicSeparableConv2d,
)
from ofa.imagenet_classification.elastic_nn.networks import OFAViT
from ofa.imagenet_classification.run_manager import (
    Cifar10RunConfig,
)
from ofa.imagenet_classification.networks import ViT
from ofa.imagenet_classification.run_manager.run_manager import (
    RunManager,
)
from ofa.utils import download_url, MyRandomResizedCrop
from ofa.imagenet_classification.elastic_nn.training.progressive_shrinking_ViT import (
    load_models,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--task",
    type=str,
    default="heads",
    choices=[
        "heads",
        "depth",
        "width_mult",
    ],
)
parser.add_argument("--phase", type=int, default=1, choices=[1, 2])
parser.add_argument("--resume", action="store_true")

args = parser.parse_args()

args.path = "exp/normal2supernet"
args.dynamic_batch_size = 2
args.n_epochs = 1000
args.base_lr = 1e-3
args.warmup_epochs = 30
args.warmup_lr = 0
args.heads_list = "12"
args.width_mult_list = "8"
args.depth_list = "20"

args.manual_seed = 0

args.lr_schedule_type = "cosine"

args.base_batch_size = 512
args.valid_size = 10000

args.opt_type = "sgd"
args.momentum = 0.9
args.no_nesterov = False
args.weight_decay = 3e-5
args.label_smoothing = 0.1
args.no_decay_keys = "bn#bias"
args.fp16_allreduce = False

args.model_init = "he_fout"
args.validation_frequency = 10
args.print_frequency = 10

args.n_worker = 8
args.resize_scale = 0.08
args.distort_color = "tf"
#args.image_size = "128,160,192,224"
args.image_size = "32"
args.continuous_size = True
args.not_sync_distributed_image_size = True

args.bn_momentum = 0.1
args.bn_eps = 1e-5
args.dropout = 0.1
args.base_stage_width = "proxyless"

args.dy_conv_scaling_mode = 1
args.independent_distributed_sampling = False

args.heads_ratio = 1.0
args.heads_type = "ce"


if __name__ == "__main__":
    os.makedirs(args.path, exist_ok=True)

    #set device
    local_rank = 0
    torch.cuda.set_device(local_rank) #rank = 0

    num_gpus = 1
    rank = 0

    args.teacher_path = None

    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)

    # image size
    args.image_size = [int(img_size) for img_size in args.image_size.split(",")]
    if len(args.image_size) == 1:
        args.image_size = args.image_size[0]
    MyRandomResizedCrop.CONTINUOUS = args.continuous_size
    MyRandomResizedCrop.SYNC_DISTRIBUTED = False
    print("image size: ", args.image_size)

    # build run config from args
    args.lr_schedule_param = None
    args.opt_param = {
        "momentum": args.momentum,
        "nesterov": not args.no_nesterov,
    }
    args.init_lr = args.base_lr
    if args.warmup_lr < 0:
        args.warmup_lr = args.base_lr
    args.train_batch_size = args.base_batch_size
    args.test_batch_size = args.base_batch_size * 4
    run_config = Cifar10RunConfig(**args.__dict__)

    # print run config information
    print("Run config:")
    for k, v in run_config.config.items():
        print("\t%s: %s" % (k, v))

    #build net from args
    args.heads_list = [int(num) for num in args.heads_list.split(", ")]
    args.width_mult_list = [int(num) for num in args.width_mult_list.split(", ")]
    args.depth_list = [int(num) for num in args.depth_list.split(", ")]

    net = OFAViT(
        n_classes=run_config.data_provider.n_classes,
        dropout_rate=args.dropout,
        image_size=args.image_size,
        dim=192,
        dim_heads=24,
        act_func="gelu",
        heads_list=args.heads_list,
        width_mult_ratio_list=args.width_mult_list,
        depth_list=args.depth_list,
    )

    """ Distributed RunManager """

    run_manager = RunManager(
        args.path,
        net,
        run_config,
    )
    run_manager.save_config()

    # training
    from ofa.imagenet_classification.elastic_nn.training.progressive_shrinking_ViT import (
        validate,
        train,
    )

    args.ofa_checkpoint_path = None #"./exp/normal2heads/checkpoint/model_best.pth.tar"

    validate_func_dict = {
        "heads_list": sorted({min(args.heads_list), max(args.heads_list)}),
        "width_mult_list": sorted({min(args.width_mult_list), max(args.width_mult_list)}),
        "depth_list": sorted({min(net.depth_list), max(net.depth_list)}),
    }
    print("validate_func_dict", validate_func_dict)

    if run_manager.start_epoch == 0:
        if args.ofa_checkpoint_path != None:
            load_models(
            run_manager,
            run_manager.net,
            args.ofa_checkpoint_path,
            )

        run_manager.write_log(
            "%.3f\t%.3f\t%.3f\t%s"
            % validate(run_manager, is_test=True, **validate_func_dict),
            "valid",
        )
    else:
        assert args.resume
    train(
        run_manager,
        args,
        lambda _run_manager, epoch, is_test: validate(
            _run_manager, epoch, is_test, **validate_func_dict
        ),
    )
