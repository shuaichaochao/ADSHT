# -*- coding: utf-8 -*-
# @Time : 2020/12/19 12:48
# @File : run.py
# @Project : VIT_Hash


import torch
import argparse
import adsh
import os

from loguru import logger
from data.data_loader import load_data
from models.modeling import VisionTransformer, CONFIGS
from models import resnet
import numpy as np


def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_%s_%s_checkpoint.bin" % (args.name, args.dataset, args.code_length))
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]" % args.output_dir)


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]

    if args.dataset == "cifar10":
        args.num_classes = 10
    elif args.dataset == "nus-wide-tc21":
        args.num_classes = 21

    # 加入映射的hash位数
    model = VisionTransformer(config, args.img_size, num_classes=args.num_classes, vis=True,
                              hash_bit=args.code_length)
    model.load_from(np.load(args.pretrained_dir))
    # model = resnet.load_model(args.num_classes, args.code_length,args.pretrained_dir)
    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s" % args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    return args, model


def run():
    args = load_config()
    # 建立日志文件
    # logger.add('logs/{time}'+args.name+'_'+args.dataset+'.log', rotation='500 MB', level='DEBUG')
    # logger.info(args)

    # 为卷积网络加速训练，使用底层优化的算法
    # torch.backends.cudnn.benchmark = True

    # Load dataset
    query_dataloader, _, retrieval_dataloader = load_data(
        args.dataset,
        args.root,
        args.num_query,
        args.num_samples,
        args.batch_size,
        args.num_workers,
    )

    for args.code_length in args.code_length_list:
        # for args.re in args.re_list:
        torch.cuda.empty_cache()
        # Model & Tokenizer Setup
        args, model = setup(args)
        mAP = adsh.train(
            model,
            query_dataloader,
            retrieval_dataloader,
            args
        )
        logger.info('[code_length:{}][map:{:.4f}]'.format(args.code_length, mAP))
        # logger.info('[code_length:{}][re:{}][map:{:.4f}]'.format(args.code_length, args.re, mAP))

        # 保存模型
        # save_model(args, model)


def load_config():
    """
    Load configuration.
    Args
        None
    Returns
        args(argparse.ArgumentParser): Configuration.
    """
    parser = argparse.ArgumentParser(description='ADSHT_PyTorch')
    parser.add_argument('--name', default='ADSHT',
                        help='model name.')

    parser.add_argument('--output_dir', default='output',
                        help='model name.')
    parser.add_argument('--dataset', default='cifar10', help='Dataset name.')
    # parser.add_argument('--root', default='./dataset', help='Path of dataset')
    parser.add_argument('--root', default='/data02/WeiHongxi/Node95/hechao/dataset', help='Path of dataset')
    # parser.add_argument('--dataset', default='nus-wide-tc21', help='Dataset name.')
    # parser.add_argument('--root', default='/data02/WeiHongxi/Node95/hechao/dataset/NUS-WIDE',
    #                     help='Path of dataset')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size.(default: 64)')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Learning rate.(default: 1e-4)')
    parser.add_argument('--code_length_list', default='24,32,48', type=str,
                        help='Binary hash code length list.(default: 24,32,48)')
    parser.add_argument('--max_iter', default=60, type=int,
                        help='Number of iterations.(default: 40)')
    parser.add_argument('--max_epoch', default=3, type=int,
                        help='Number of epochs.(default: 3)')
    parser.add_argument('--num_query', default=1000, type=int,
                        help='Number of query data points.(default: 1000)')
    parser.add_argument('--num_samples', default=5000, type=int,
                        help='Number of sampling data points.(default: 5000)')
    # parser.add_argument('--num_query', default=2100, type=int,
    #                     help='Number of query data points.(default: 1000)')
    # parser.add_argument('--num_samples', default=2100, type=int,
    #                     help='Number of sampling data points.(default: 5000)')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of loading data threads.(default: 0)')
    parser.add_argument('--topk', default=-1, type=int,
                        help='Calculate map of top k.(default: all)')
    parser.add_argument('--gamma', default=200, type=float,
                        help='Hyper-parameter,quantization_loss(default: 200)')
    parser.add_argument('--re_list', default='0.05,0.1,0.2,1,5,10,20', type=str,
                        help='Hyper-parameter,Classified loss list(retrieval dataset)(default: 10)')
    parser.add_argument('--re', default=10, type=float,
                        help='Hyper-parameter,Classified loss(retrieval dataset)(default: 10)')
    parser.add_argument('--v_list', default='0.5,1,1.5,2,5', type=str,
                        help='Hyper-parameter,regular_loss list(default: 1)')
    parser.add_argument('--v', default=1, type=float,
                        help='Hyper-parameter,regular_loss (default: 1)')
    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument("--pretrained_dir", type=str, default="checkpoint/ViT-L_32_image.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--model_type", type=str, default="ViT-L_32",
                        help="Which variant to use.")
    # parser.add_argument("--pretrained_dir", type=str, default="/home/WeiHongxi/Node95/hechao/datacode/hash/VIT_Hash_MM/checkpoint/resnet50.pth",
    #                     help="Where to search for pretrained ViT models.")
    parser.add_argument('--fp16', action='store_true', default=True,
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    args = parser.parse_args()

    if torch.cuda.is_available():
        args.device = torch.device("cuda:0")
    else:
        args.device = torch.device("cpu")

    # Hash code length
    args.code_length_list = list(map(int, args.code_length_list.split(',')))

    # Hyper-parameter,Classified loss list(retrieval dataset)
    args.re_list = list(map(float, args.re_list.split(',')))

    # Hyper-parameter,regular_loss list
    # args.v_list = list(map(float, args.v_list.split(',')))

    return args


if __name__ == '__main__':
    run()
