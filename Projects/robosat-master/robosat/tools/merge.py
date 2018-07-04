import sys
import argparse

import geojson

import shapely.geometry

from robosat.spatial.core import make_index, project, union
from robosat.graph.core import UndirectedGraph


def add_parser(subparser):
    parser = subparser.add_parser('merge', help='merged adjacent GeoJSON features',
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('features', type=str, help='GeoJSON file to read features from')
    parser.add_argument('--threshold', type=int, required=True, help='minimum distance to adjacent features, in m')
    parser.add_argument('out', type=str, help='path to GeoJSON to save merged features to')

    parser.set_defaults(func=main)


def main(args):
    with open(args.features) as fp:
        collection = geojson.load(fp)

    shapes = [shapely.geometry.shape(feature['geometry']) for feature in collection['features']]
    del collection

    graph = UndirectedGraph()
    idx = make_index(shapes)

    def buffered(shape):
        projected = project(shape, 'epsg:4326', 'epsg:3395')
        buffered = projected.buffer(args.threshold)
        unprojected = project(buffered, 'epsg:3395', 'epsg:4326')
        return unprojected

    def unbuffered(shape):
        projected = project(shape, 'epsg:4326', 'epsg:3395')
        unbuffered = projected.buffer(-1 * args.threshold)
        unprojected = project(unbuffered, 'epsg:3395', 'epsg:4326')
        return unprojected

    for i, shape in enumerate(shapes):
        embiggened = buffered(shape)

        graph.add_edge(i, i)

        nearest = [j for j in idx.intersection(embiggened.bounds, objects=False) if i != j]

        for t in nearest:
            if embiggened.intersects(shapes[t]):
                graph.add_edge(i, t)

    components = list(graph.components())
    assert sum([len(v) for v in components]) == len(shapes), 'components capture all shape indices'
    print('Merged {} features into {} features'.format(len(shapes), len(components)), file=sys.stderr)

    features = []

    for component in components:
        embiggened = [buffered(shapes[v]) for v in component]
        merged = unbuffered(union(embiggened))

        if merged.is_valid:
            feature = geojson.Feature(geometry=shapely.geometry.mapping(merged))
            features.append(feature)
        else:
            print('Warning: merged feature is not valid, skipping', file=sys.stderr)

    collection = geojson.FeatureCollection(features)

    with open(args.out, 'w') as fp:
        geojson.dump(collection, fp)
