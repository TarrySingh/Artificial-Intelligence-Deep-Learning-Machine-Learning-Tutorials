import sys

import osmium
import geojson
import shapely.geometry


class BuildingHandler(osmium.SimpleHandler):
    '''Extracts building polygon features (visible in satellite imagery) from the map.
    '''

    # building=* to discard because these features are not vislible in satellite imagery
    building_filter = set(['construction', 'houseboat', 'static_caravan', 'stadium',
                           'conservatory', 'digester', 'greenhouse', 'ruins'])

    def __init__(self):
        super().__init__()
        self.features = []

    def way(self, w):
        if not w.is_closed():
            return

        if 'building' not in w.tags:
            return

        if w.tags['building'] in self.building_filter:
            return

        geometry = geojson.Polygon([[(n.lon, n.lat) for n in w.nodes]])
        shape = shapely.geometry.shape(geometry)

        if shape.is_valid:
            feature = geojson.Feature(geometry=geometry)
            self.features.append(feature)
        else:
            print('Warning: invalid feature: https://www.openstreetmap.org/way/{}'.format(w.id), file=sys.stderr)

    def save(self, out):
        collection = geojson.FeatureCollection(self.features)

        with open(out, 'w') as fp:
            geojson.dump(collection, fp)
