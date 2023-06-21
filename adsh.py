# -*- coding: utf-8 -*-
# @Time : 2020/12/19 12:49
# @File : adsh.py
# @Project : VIT_Hash
import torch
import torch.optim as optim
import os
import time
# import models.alexnet as alexnet
# import models.resnet as resnet
import utils.evaluate as evaluate

from loguru import logger
from models.adsh_loss import ADSH_Loss
from data.data_loader import sample_dataloader
from apex import amp


def train(
        model,
        query_dataloader,
        retrieval_dataloader,
        args
):
    """
    Training model.
    Args
        query_dataloader, retrieval_dataloader(torch.utils.data.dataloader.DataLoader): Data loader.
        code_length(int): Hashing code length.
        device(torch.device): GPU or CPU.
        lr(float): Learning rate.
        max_iter(int): Number of iterations.
        max_epoch(int): Number of epochs.
        num_train(int): Number of sampling training data points.
        batch_size(int): Batch size.
        root(str): Path of dataset.
        dataset(str): Dataset name.
        gamma(float): Hyper-parameters.
        topk(int): Topk k map.
    Returns
        mAP(float): Mean Average Precision.
    """
    # Initialization
    # model = alexnet.load_model(args.code_length).to(args.device)
    # model = resnet.load_model(10, args.code_length).to(args.device)
    # optimizer = optim.Adam(
    #     model.parameters(),
    #     lr=args.lr,
    #     weight_decay=1e-5,
    # )

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-5,
    )

    if args.fp16:
        model, optimizer = amp.initialize(models=model,
                                          optimizers=optimizer,
                                          opt_level=args.fp16_opt_level)
        amp._amp_state.loss_scalers[0]._loss_scale = 2 ** 20

    criterion = ADSH_Loss(args.code_length, args.gamma)

    num_retrieval = len(retrieval_dataloader.dataset)

    # W = torch.zeros(args.code_length, args.num_classes).to(args.device)
    U = torch.zeros(args.num_samples, args.code_length).to(args.device)
    B = torch.randn(num_retrieval, args.code_length).to(args.device)
    retrieval_targets = retrieval_dataloader.dataset.get_onehot_targets().to(args.device)
    # Label information
    Y = retrieval_targets

    start = time.time()
    for it in range(args.max_iter):
        iter_start = time.time()
        # Sample training data for cnn learning
        train_dataloader, sample_index = sample_dataloader(retrieval_dataloader, args.num_samples, args.batch_size,
                                                           args.root, args.dataset,args.num_workers)

        # Create Similarity matrix
        train_targets = train_dataloader.dataset.get_onehot_targets().to(args.device)
        # train_targets = torch.where(train_targets == 1, torch.full_like(train_targets, 1), torch.full_like(
        # train_targets, -1))
        # retrieval_targets = torch.where(Y == 1, torch.full_like(Y, 1), torch.full_like(Y, -1))

        # pytorch中@是用来进行矩阵相乘的
        S = (train_targets @ retrieval_targets.t() > 0).float()
        # S = torch.where(S == 1, torch.full_like(S, 1), torch.full_like(S, -1))
        S = torch.where(S == 1, S, torch.full_like(S, -1))

        # Soft similarity matrix, benefit to converge
        r = S.sum() / (1 - S).sum()
        S = S * (1 + r) - r

        # Training CNN model
        for epoch in range(args.max_epoch):
            for batch, (data, targets, index) in enumerate(train_dataloader):
                data, targets, index = data.to(args.device), targets.to(args.device), index.to(args.device)
                optimizer.zero_grad()
                # model.zero_grad()

                F = model(data)
                U[index, :] = F.data

                total_loss = criterion(F, B, S[index, :], sample_index[index])


                if args.fp16:
                    with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    total_loss.backward()
                optimizer.step()

                # # # Update W
                # W = torch.inverse(
                #     B.t() @ B + args.v / args.re * torch.eye(args.code_length, device=args.device)) @ B.t() @ Y

        # # Update W
        # W = torch.inverse(B.t() @ B + args.v / args.re * torch.eye(args.code_length, device=args.device)) @ B.t() @ Y

        # Update B  这里修改过，后移一个tab键
        expand_U = torch.zeros(B.shape).to(args.device)
        expand_U[sample_index, :] = U

        B = solve_dcc(Y, B, U, S, expand_U, args.code_length, args.gamma)
        # B = solve_dcc(W, Y, B, U, S, args.code_length, args.re)

        # torch.cuda.empty_cache()
        # # Total loss
        # iter_loss = calc_loss(W, Y, U, B, S, args.code_length, sample_index, args.gamma, args.re, args.v)
        iter_loss = calc_loss(Y, U, B, S, args.code_length, sample_index, args.gamma, args.re, args.v)
        logger.debug('[iter:{}/{}][loss:{:.2f}][iter_time:{:.2f}]'.format(it+1, args.max_iter, iter_loss,
                                                                          time.time() - iter_start))
    logger.info('[Training time:{:.2f}]'.format(time.time()-start))

    # Evaluate
    query_code = generate_code(model, query_dataloader, args.code_length, args.device)
    mAP = evaluate.mean_average_precision(
        query_code.to(args.device),
        B,
        query_dataloader.dataset.get_onehot_targets().to(args.device),
        retrieval_targets,
        args.device,
        args.topk,
    )

    # Save checkpoints
    # torch.save(query_code.cpu(), os.path.join(args.output_dir, '%s_%s_%s_query_code.t' % (args.name, args.dataset, args.code_length)))
    # torch.save(B.cpu(), os.path.join(args.output_dir, '%s_%s_%s_database_code.t' % (args.name, args.dataset, args.code_length)))
    # torch.save(query_dataloader.dataset.get_onehot_targets, os.path.join(args.output_dir, '%s_%s_%s_query_targets.t' % (args.name, args.dataset, args.code_length)))
    # torch.save(retrieval_targets.cpu(), os.path.join(args.output_dir, '%s_%s_%s_database_targets.t' % (args.name, args.dataset, args.code_length)))

    return mAP

def solve_dcc(Y, B, U, S, expand_U, code_length, gamma):
# def solve_dcc(W, Y, B, U, S, code_length, re):
    """
    Solve DCC problem.
    """
    Q = (code_length * S).t() @ U + gamma * expand_U
    # Q = (code_length * S).t() @ U + re * Y @ W.t()
    for bit in range(code_length):
        q = Q[:, bit]
        u = U[:, bit]
        # w = W[bit, :]
        B_prime = torch.cat((B[:, :bit], B[:, bit+1:]), dim=1)
        U_prime = torch.cat((U[:, :bit], U[:, bit+1:]), dim=1)
        # W_prime = torch.cat((W[:bit, :], W[bit + 1:, :]), dim=0)

        # B[:, bit] = (q.t() - B_prime @ U_prime.t() @ u - re *B_prime @ W_prime @ w.t()).sign()
        B[:, bit] = (q.t() - B_prime @ U_prime.t() @ u).sign()

    return B


def calc_loss(Y, U, B, S, code_length, sample_index, gamma, re, v):
# def calc_loss(W, Y, U, B, S, code_length, sample_index, gamma, re, v):
    """
    Calculate loss.
    """
    hash_loss = ((code_length * S - U @ B.t()).pow(2)).sum()
    quantization_loss = ((U - B[sample_index, :]) ** 2).sum()
    # # 添加分类损失
    # logits_loss = ((Y - B @ W) ** 2).sum() / Y.shape[0]
    # # 权重规范化损失
    # regular_loss = (W ** 2).sum() / W.shape[0]
    # loss = (hash_loss + gamma * quantization_loss) / (U.shape[0] * B.shape[0]) + re*logits_loss + v*regular_loss
    loss = (hash_loss + gamma * quantization_loss) / (U.shape[0] * B.shape[0])
    return loss.item()


def generate_code(model, dataloader, code_length, device):
    """
    Generate hash code
    Args
        dataloader(torch.utils.data.DataLoader): Data loader.
        code_length(int): Hash code length.
        device(torch.device): Using gpu or cpu.
    Returns
        code(torch.Tensor): Hash code.
    """
    model.eval()
    with torch.no_grad():
        N = len(dataloader.dataset)
        code = torch.zeros([N, code_length])
        for data, _, index in dataloader:
            data = data.to(device)
            # hash_code, _ = model(data)
            hash_code = model(data)
            code[index, :] = hash_code.sign().cpu()

    model.train()
    return code
