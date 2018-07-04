import sys

import osmium
import geojson
import shapely.geometry


class ParkingHandler(osmium.SimpleHandler):
    '''Extracts parking lot polygon features (visible in satellite imagery) from the map.
    '''

    # parking=* to discard because these features are not vislible in satellite imagery
    parking_filter = set(['underground', 'sheds', 'carports', 'garage_boxes'])

    def __init__(self):
        super().__init__()
        self.features = []

    def way(self, w):
        if not w.is_closed():
            return

        if 'amenity' not in w.tags or w.tags['amenity'] != 'parking':
            return

        if 'parking' in w.tags:
            if w.tags['parking'] in self.parking_filter:
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
