import argparse
import os
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize

from tqdm import tqdm
from PIL import Image

from robosat.datasets import BufferedSlippyMapDirectory
from robosat.unet import UNet
from robosat.config import load_config
from robosat.colors import continuous_palette_for_color
from robosat.transforms import ConvertImageMode, ImageToTensor


def add_parser(subparser):
    parser = subparser.add_parser('predict', help='predicts probability masks for slippy map tiles',
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--batch_size', type=int, default=1, help='images per batch')
    parser.add_argument('--checkpoint', type=str, required=True, help='model checkpoint to load')
    parser.add_argument('--overlap', type=int, default=32, help='tile pixel overlap to predict on')
    parser.add_argument('--tile_size', type=int, default=512, help='tile size for slippy map tiles')
    parser.add_argument('--workers', type=int, default=1, help='number of workers pre-processing images')
    parser.add_argument('tiles', type=str, help='directory to read slippy map image tiles from')
    parser.add_argument('probs', type=str, help='directory to save slippy map probability masks to')
    parser.add_argument('--model', type=str, required=True, help='path to model configuration file')
    parser.add_argument('--dataset', type=str, required=True, help='path to dataset configuration file')

    parser.set_defaults(func=main)


def main(args):
    model = load_config(args.model)
    dataset = load_config(args.dataset)

    cuda = model['common']['cuda']

    device = torch.device('cuda' if cuda else 'cpu')

    def map_location(storage, _):
        return storage.cuda() if cuda else storage.cpu()

    if cuda and not torch.cuda.is_available():
        sys.exit('Error: CUDA requested but not available')

    num_classes = len(dataset['common']['classes'])

    # https://github.com/pytorch/pytorch/issues/7178
    chkpt = torch.load(args.checkpoint, map_location=map_location)

    net = UNet(num_classes).to(device)
    net = nn.DataParallel(net)

    if cuda:
        torch.backends.cudnn.benchmark = True

    net.load_state_dict(chkpt)
    net.eval()

    transform = Compose([
        ConvertImageMode(mode='RGB'),
        ImageToTensor(),
        Normalize(mean=dataset['stats']['mean'], std=dataset['stats']['std'])
    ])

    directory = BufferedSlippyMapDirectory(args.tiles, transform=transform, size=args.tile_size, overlap=args.overlap)
    loader = DataLoader(directory, batch_size=args.batch_size)

    # don't track tensors with autograd during prediction
    with torch.no_grad():
        for images, tiles in tqdm(loader, desc='Eval', unit='batch', ascii=True):
            images = images.to(device)
            outputs = net(images)

            # manually compute segmentation mask class probabilities per pixel
            probs = nn.functional.softmax(outputs, dim=1).data.cpu().numpy()

            for tile, prob in zip(tiles, probs):
                x, y, z = list(map(int, tile))

                # we predicted on buffered tiles; now get back probs for original image
                prob = directory.unbuffer(prob)

                # Quantize the floating point probabilities in [0,1] to [0,255] and store
                # a single-channel `.png` file with a continuous color palette attached.

                assert prob.shape[0] == 2, 'single channel requires binary model'
                assert np.allclose(np.sum(prob, axis=0), 1.), 'single channel requires probabilities to sum up to one'
                foreground = prob[1:, :, :]

                anchors = np.linspace(0, 1, 256)
                quantized = np.digitize(foreground, anchors).astype(np.uint8)

                palette = continuous_palette_for_color('pink', 256)

                out = Image.fromarray(quantized.squeeze(), mode='P')
                out.putpalette(palette)

                os.makedirs(os.path.join(args.probs, str(z), str(x)), exist_ok=True)
                path = os.path.join(args.probs, str(z), str(x), str(y) + '.png')

                out.save(path, optimize=True)
