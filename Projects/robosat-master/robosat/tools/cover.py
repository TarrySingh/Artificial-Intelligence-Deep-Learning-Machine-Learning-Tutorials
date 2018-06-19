import argparse
import csv
import json

from supermercado import burntiles
from tqdm import tqdm


def add_parser(subparser):
    parser = subparser.add_parser('cover', help='generates tiles covering GeoJSON features',
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--zoom', type=int, required=True, help='zoom level of tiles')
    parser.add_argument('features', type=str, help='path to GeoJSON features')
    parser.add_argument('out', type=str, help='path to csv file to store tiles in')

    parser.set_defaults(func=main)


def main(args):
    with open(args.features) as f:
        features = json.load(f)

    tiles = []

    for feature in tqdm(features['features'], ascii=True, unit='feature'):
        tiles.extend(map(tuple, burntiles.burn([feature], args.zoom).tolist()))

    # tiles can overlap for multiple features; unique tile ids
    tiles = list(set(tiles))

    with open(args.out, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerows(tiles)
