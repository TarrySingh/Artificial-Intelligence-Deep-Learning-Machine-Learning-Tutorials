import os
import argparse

from PIL import Image
from tqdm import tqdm
import numpy as np

from robosat.tiles import tiles_from_slippy_map


def add_parser(subparser):
    parser = subparser.add_parser('compare', help='compare images, labels and masks side by side',
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('out', type=str, help='directory to save visualizations to')
    parser.add_argument('images', type=str, help='directory to read slippy map images from')
    parser.add_argument('labels', type=str, help='directory to read slippy map labels from')
    parser.add_argument('masks', type=str, nargs='+', help='slippy map directories to read masks from')
    parser.add_argument('--minimum', type=float, default=0.0, help='minimum percentage of mask not background')
    parser.add_argument('--maximum', type=float, default=1.0, help='maximum percentage of mask not background')

    parser.set_defaults(func=main)


def main(args):
    images = tiles_from_slippy_map(args.images)

    for tile, path in tqdm(list(images), desc='Compare', unit='image', ascii=True):
        x, y, z = list(map(str, tile))

        image = Image.open(path).convert('RGB')
        label = Image.open(os.path.join(args.labels, z, x, '{}.png'.format(y))).convert('P')
        assert image.size == label.size

        keep = False
        masks = []
        for path in args.masks:
            mask = Image.open(os.path.join(path, z, x, '{}.png'.format(y))).convert('P')
            assert image.size == mask.size
            masks.append(mask)

            # TODO: The calculation below does not work for multi-class.
            percentage = np.sum(np.array(mask) != 0) / np.prod(image.size)

            # Keep this image when percentage is within required threshold.
            if percentage >= args.minimum and percentage <= args.maximum:
                keep = True

        if not keep:
            continue

        width, height = image.size

        # Columns for image, label and all the masks.
        columns = 2 + len(masks)
        combined = Image.new(mode='RGB', size=(columns * width, height))

        combined.paste(image, box=(0 * width, 0))
        combined.paste(label, box=(1 * width, 0))
        for i, mask in enumerate(masks):
            combined.paste(mask, box=((2 + i) * width, 0))

        os.makedirs(os.path.join(args.out, z, x), exist_ok=True)
        path = os.path.join(args.out, z, x, '{}.png'.format(y))
        combined.save(path, optimize=True)
