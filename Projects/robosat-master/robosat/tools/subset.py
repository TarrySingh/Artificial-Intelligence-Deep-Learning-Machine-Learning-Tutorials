import os
import argparse
import shutil

from tqdm import tqdm

from robosat.tiles import tiles_from_slippy_map, tiles_from_csv


def add_parser(subparser):
    parser = subparser.add_parser('subset', help='filter images in a slippy map directory using a csv',
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('images', type=str, help='directory to read slippy map image tiles from for filtering')
    parser.add_argument('tiles', type=str, help='csv to filter images by')
    parser.add_argument('out', type=str, help='directory to save filtered images to')

    parser.set_defaults(func=main)


def main(args):
    images = tiles_from_slippy_map(args.images)

    tiles = set(tiles_from_csv(args.tiles))

    for tile, src in tqdm(list(images), desc='Subset', unit='image', ascii=True):
        if tile not in tiles:
            continue

        # The extention also includes the period.
        extention = os.path.splitext(src)[1]

        os.makedirs(os.path.join(args.out, str(tile.z), str(tile.x)), exist_ok=True)
        dst = os.path.join(args.out, str(tile.z), str(tile.x), '{}{}'.format(tile.y, extention))

        shutil.copyfile(src, dst)
