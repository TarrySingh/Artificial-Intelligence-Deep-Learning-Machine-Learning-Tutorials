import json
import argparse
import functools

import geojson
from tqdm import tqdm

import shapely.geometry

from robosat.spatial.core import make_index, iou


def add_parser(subparser):
    parser = subparser.add_parser('dedupe', help='deduplicates features against OpenStreetMap',
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('osm', type=str, help='ground truth GeoJSON feature collection from OpenStreetMap')
    parser.add_argument('predicted', type=str, help='predicted GeoJSON feature collection to deduplicate')
    parser.add_argument('--threshold', type=float, required=True,
                        help='maximum allowed IoU to keep predictions, between 0.0 and 1.0')
    parser.add_argument('out', type=str, help='path to GeoJSON to save deduplicated features to')

    parser.set_defaults(func=main)


def main(args):
    with open(args.osm) as fp:
        osm = json.load(fp)

    # Todo: at the moment we load all OSM shapes. It would be more efficient to tile
    #       cover and load only OSM shapes in the tiles covering the predicted shapes.
    osm_shapes = [shapely.geometry.shape(feature['geometry']) for feature in osm['features']]
    del osm

    with open(args.predicted) as fp:
        predicted = json.load(fp)

    predicted_shapes = [shapely.geometry.shape(features['geometry']) for features in predicted['features']]
    del predicted

    idx = make_index(osm_shapes)
    features = []

    for predicted_shape in tqdm(predicted_shapes, desc='Dedupe', unit='image', ascii=True):
        nearby = [osm_shapes[i] for i in idx.intersection(predicted_shape.bounds, objects=False)]

        keep = False

        if not nearby:
            keep = True
        else:
            intersecting = [shape for shape in nearby if predicted_shape.intersects(shape)]

            if not intersecting:
                keep = True
            else:
                intersecting_shapes = functools.reduce(lambda lhs, rhs: lhs.union(rhs), intersecting)

                if iou(predicted_shape, intersecting_shapes) < args.threshold:
                    keep = True

        if keep:
            feature = geojson.Feature(geometry=shapely.geometry.mapping(predicted_shape))
            features.append(feature)

    collection = geojson.FeatureCollection(features)

    with open(args.out, 'w') as fp:
        geojson.dump(collection, fp)
