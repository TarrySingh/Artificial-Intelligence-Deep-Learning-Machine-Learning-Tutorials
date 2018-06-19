import os
import sys
import argparse

import numpy as np

from PIL import Image

from robosat.tiles import tiles_from_slippy_map
from robosat.colors import make_palette


def add_parser(subparser):
    parser = subparser.add_parser('masks', help='compute masks from prediction probabilities',
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('masks', type=str, help='slippy map directory to save masks to')
    parser.add_argument('probs', type=str, nargs='+', help='slippy map directories with class probabilities')
    parser.add_argument('--weights', type=float, nargs='+', help='weights for weighted average soft-voting')

    parser.set_defaults(func=main)


def main(args):
    if args.weights and len(args.probs) != len(args.weights):
        sys.exit('Error: number of slippy map directories and weights must be the same')

    tilesets = map(tiles_from_slippy_map, args.probs)

    for tileset in zip(*tilesets):
        tiles = [tile for tile, _ in tileset]
        paths = [path for _, path in tileset]

        assert len(set(tiles)), 'tilesets in sync'
        x, y, z = tiles[0]

        # Un-quantize the probabilities in [0,255] to floating point values in [0,1]
        anchors = np.linspace(0, 1, 256)

        def load(path):
            # Note: assumes binary case and probability sums up to one.
            # Needs to be in sync with how we store them in prediction.

            quantized = np.array(Image.open(path).convert('P'))

            # (512, 512, 1) -> (1, 512, 512)
            foreground = np.rollaxis(np.expand_dims(anchors[quantized], axis=0), axis=0)
            background = np.rollaxis(1. - foreground, axis=0)

            # (1, 512, 512) + (1, 512, 512) -> (2, 512, 512)
            return np.concatenate((background, foreground), axis=0)

        probs = [load(path) for path in paths]

        mask = softvote(probs, axis=0, weights=args.weights)
        mask = mask.astype(np.uint8)

        palette = make_palette('denim', 'orange')
        out = Image.fromarray(mask, mode='P')
        out.putpalette(palette)

        os.makedirs(os.path.join(args.masks, str(z), str(x)), exist_ok=True)

        path = os.path.join(args.masks, str(z), str(x), str(y) + '.png')
        out.save(path, optimize=True)


def softvote(probs, axis=0, weights=None):
    '''Weighted average soft-voting to transform class probabilities into class indices.

    Args:
      probs: array-like probabilities to average.
      axis: axis or axes along which to soft-vote.
      weights: array-like for weighting probabilities.

    Notes:
      See http://scikit-learn.org/stable/modules/ensemble.html#weighted-average-probabilities-soft-voting
    '''

    return np.argmax(np.average(probs, axis=axis, weights=weights), axis=axis)
