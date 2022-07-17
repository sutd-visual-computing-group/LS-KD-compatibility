"""
Create train, valid, test iterators for CIFAR-10 [1].
Easily extended to MNIST, CIFAR-100 and Imagenet.

Reference:
[1] https://discuss.pytorch.org/t/feedback-on-pytorch-for-kaggle-competitions/2252/4
[2] https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb#file-data_loader-py
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from plot_cifar import plot_images
from torchvision import datasets
from torchvision.datasets import ImageFolder

from torchvision import transforms
from torch.utils.data.sampler import Sampler
from torch.utils.data.sampler import RandomSampler

from torch._six import int_classes as _int_classes
from torch import Tensor

from typing import Iterator, Optional, Sequence, List, TypeVar, Generic, Sized
T_co = TypeVar('T_co', covariant=True)

# cifar-10 statistics
MEAN = {
    'cifar10': [0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
    'cifar100': [0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
    'easy_imagenet': [0.485, 0.456, 0.406],
}

# cifar-10 statistics
STD = {
    'cifar10': [0.2673342858792401, 0.2564384629170883, 0.27615047132568404],
    'cifar100': [0.2673342858792401, 0.2564384629170883, 0.27615047132568404],
    'easy_imagenet': [0.229, 0.224, 0.225],
}


# Custom sampler
class SubsetSequentialSampler(Sampler[int]):
    r"""Samples elements sequentially from a given list of indices, without replacement.

    Args:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """
    indices: Sequence[int]

    def __init__(self, indices: Sequence[int], data_source: Optional[Sized], generator=None) -> None:
        super().__init__(data_source)
        self.indices = indices
        self.generator = generator

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


def get_train_valid_loader(data_dir,
                           batch_size,
                           augment,
                           shuffle=False,
                           show_sample=False,
                           num_workers=1,
                           pin_memory=True):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the subset of ImageNet dataset. A sample
    9x9 grid of the images can be optionally displayed.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.

    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # # define transforms

    if augment:  # data augmentation on training set
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        valid_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    # load the dataset
    train_dataset = ImageFolder(
        root=data_dir+'/train', transform=train_transform,
    )

    valid_dataset = ImageFolder(
        root=data_dir+'/val', transform=valid_transform,
    )

    num_train = len(train_dataset)
    num_valid = len(valid_dataset)
    train_idx = list(range(num_train))
    valid_idx = list(range(num_valid))

    # instead we apply the sequential sampler for our visualization
    # train_sampler = RandomSampler(train_idx)
    # valid_sampler = RandomSampler(valid_idx)
    train_sampler = SubsetSequentialSampler(train_idx, data_source=train_dataset)
    valid_sampler = SubsetSequentialSampler(valid_idx, data_source=valid_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    # visualize some images, can also be used to plot other images
    if show_sample:
        sample_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=9, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        data_iter = iter(sample_loader)
        images, labels = data_iter.next()
        X = images.numpy().transpose([0, 2, 3, 1])
        plot_images(X, labels)

    return (train_loader, valid_loader)


def get_test_loader(data_dir,
                    batch_size,
                    shuffle=True,
                    num_workers=1,
                    pin_memory=True):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR-10 dataset.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - shuffle: whether to shuffle the dataset after every epoch.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.

    Returns
    -------
    - data_loader: test set iterator.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    dataset = datasets.CIFAR10(
        root=data_dir, train=False,
        download=True, transform=transform,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, target, smoothing=0.3):
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = 3 * confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()
