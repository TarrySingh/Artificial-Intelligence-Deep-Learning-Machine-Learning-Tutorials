import os
import argparse

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from robosat.config import load_config
from robosat.datasets import SlippyMapTiles
from robosat.transforms import ConvertImageMode, ImageToTensor


def add_parser(subparser):
    parser = subparser.add_parser('stats', help='computes mean and std dev on dataset',
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, required=True, help='path to dataset configuration file')

    parser.set_defaults(func=main)


def main(args):
    dataset = load_config(args.dataset)
    path = dataset['common']['dataset']

    train_transform = Compose([
        ConvertImageMode(mode='RGB'),
        ImageToTensor()
    ])

    train_dataset = SlippyMapTiles(os.path.join(path, 'training', 'images'), transform=train_transform)

    n = 0
    mean = np.zeros(3, dtype=np.float64)

    loader = DataLoader(train_dataset, batch_size=1)
    for images, tile in tqdm(loader, desc='Loading', unit='image', ascii=True):
        image = torch.squeeze(images)
        assert image.size(0) == 3, 'channel first'

        image = np.array(image, dtype=np.float64)
        n += image.shape[1] * image.shape[2]

        mean += np.sum(image, axis=(1, 2))

    mean /= n
    mean.round(decimals=6, out=mean)
    print('mean: {}'.format(mean.tolist()))

    std = np.zeros(3, dtype=np.float64)

    loader = DataLoader(train_dataset, batch_size=1)
    for images, tile in tqdm(loader, desc='Loading', unit='image', ascii=True):
        image = torch.squeeze(images)
        assert image.size(0) == 3, 'channel first'

        image = np.array(image, dtype=np.float64)
        difference = np.transpose(image, (1, 2, 0)) - mean
        std += np.sum(np.square(difference), axis=(0, 1))

    std = np.sqrt(std / (n - 1))
    std.round(decimals=6, out=std)
    print('std: {}'.format(std.tolist()))
