# -*- coding: utf-8 -*-
# @Time : 2020/12/19 12:51
# @File : cifar100.py
# @Project : VIT_Hash

import torch
import numpy as np
from PIL import Image
import os
import sys
import pickle

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from data.transform import train_transform, query_transform, Onehot, encode_onehot


def load_data(root, num_query, num_train, batch_size, num_workers):
    """
    Load cifar10 dataset.
    Args
        root(str): Path of dataset.
        num_query(int): Number of query data points.
        num_train(int): Number of training data points.
        batch_size(int): Batch size.
        num_workers(int): Number of loading data threads.
    Returns
        query_dataloader, train_dataloader, retrieval_dataloader(torch.evaluate.data.DataLoader): Data loader.
    """
    CIFAR100.init(root, num_query, num_train)
    query_dataset = CIFAR100('query', transform=query_transform(), target_transform=Onehot())
    train_dataset = CIFAR100('train', transform=train_transform(), target_transform=None)
    retrieval_dataset = CIFAR100('database', transform=query_transform(), target_transform=Onehot())

    query_dataloader = DataLoader(
        query_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
      )
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
      )
    retrieval_dataloader = DataLoader(
        retrieval_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
    )

    return query_dataloader, train_dataloader, retrieval_dataloader


class CIFAR100(Dataset):
    """
    Cifar100 dataset.
    """
    @staticmethod
    def init(root, num_query, num_train):
        data_list = ['train',
                     'test',
                     ]
        base_folder = 'cifar-100-python'

        data = []
        targets = []

        for file_name in data_list:
            file_path = os.path.join(root, base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                data.append(entry['data'])
                if 'labels' in entry:
                    targets.extend(entry['labels'])
                else:
                    targets.extend(entry['fine_labels'])

        # 按行顺序堆叠
        data = np.vstack(data).reshape(-1, 3, 32, 32)
        data = data.transpose((0, 2, 3, 1))  # convert to HWC
        targets = np.array(targets)

        # Sort by class
        sort_index = targets.argsort()
        data = data[sort_index, :]
        targets = targets[sort_index]

        # (num_query / number of class) query images per class
        # (num_train / number of class) train images per class
        query_per_class = num_query // 100
        train_per_class = num_train // 100

        # Permutate index (range 0 - 600 per class) 随机排序
        perm_index = np.random.permutation(data.shape[0] // 100)
        query_index = perm_index[:query_per_class]
        train_index = perm_index[query_per_class: query_per_class + train_per_class]

        # 将query_index和train_index复制一百次，拼接
        query_index = np.tile(query_index, 100)
        train_index = np.tile(train_index, 100)

        # 将序号0-600-1200以此类推
        inc_index = np.array([i * (data.shape[0] // 100) for i in range(100)])
        # 重复建立成10个0,10个600以此再加上原query_index，保证每个类取10，按标号来的
        query_index = query_index + inc_index.repeat(query_per_class)
        train_index = train_index + inc_index.repeat(train_per_class)
        list_query_index = [i for i in query_index]
        retrieval_index = np.array(list(set(range(data.shape[0])) - set(list_query_index)), dtype=np.int)

        # Split data, targets
        CIFAR100.QUERY_IMG = data[query_index, :]
        CIFAR100.QUERY_TARGET = targets[query_index]
        CIFAR100.TRAIN_IMG = data[train_index, :]
        CIFAR100.TRAIN_TARGET = targets[train_index]
        CIFAR100.RETRIEVAL_IMG = data[retrieval_index, :]
        CIFAR100.RETRIEVAL_TARGET = targets[retrieval_index]

    def __init__(self, mode='train',
                 transform=None, target_transform=None,
                 ):
        self.transform = transform
        self.target_transform = target_transform

        if mode == 'train':
            self.data = CIFAR100.TRAIN_IMG
            self.targets = CIFAR100.TRAIN_TARGET
        elif mode == 'query':
            self.data = CIFAR100.QUERY_IMG
            self.targets = CIFAR100.QUERY_TARGET
        else:
            self.data = CIFAR100.RETRIEVAL_IMG
            self.targets = CIFAR100.RETRIEVAL_TARGET

        self.onehot_targets = encode_onehot(self.targets, 100)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, index) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.data)

    def get_onehot_targets(self):
        """
        Return one-hot encoding targets.
        """
        return torch.from_numpy(self.onehot_targets).float()
