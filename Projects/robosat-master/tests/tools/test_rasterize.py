import json
import unittest

import numpy as np
import mercantile

from robosat.tools.rasterize import burn, feature_to_mercator


def get_parking():
    with open('tests/fixtures/parking/features.geojson') as f:
        parking_fc = json.load(f)

    assert len(parking_fc['features']) == 2
    return parking_fc


class TestBurn(unittest.TestCase):

    def test_burn_with_feature(self):
        parking_fc = get_parking()

        # The tile below has a parking lot in our fixtures.
        tile = mercantile.Tile(70762, 104119, 18)

        rasterized = burn(tile, parking_fc['features'], 512)

        # rasterized.save('rasterized.png')

        self.assertEqual(rasterized.size, (512, 512))

        # Tile has a parking feature in our fixtures, thus sum should be non-zero.
        self.assertNotEqual(np.sum(rasterized), 0)

    def test_burn_without_feature(self):
        parking_fc = get_parking()

        # This tile does not have a parking lot in our fixtures.
        tile = mercantile.Tile(69623, 104946, 18)

        rasterized = burn(tile, parking_fc['features'], 512)

        self.assertEqual(rasterized.size, (512, 512))

        # Tile does not have a parking feature in our fixture, the sum of pixels is zero.
        self.assertEqual(np.sum(rasterized), 0)


class TestFeatureToMercator(unittest.TestCase):

    def test_feature_to_mercator(self):
        parking_fc = get_parking()

        parking = parking_fc['features'][0]
        mercator = next(feature_to_mercator(parking))

        self.assertEqual(mercator['type'], 'Polygon')
        self.assertEqual(int(mercator['coordinates'][0][0][0]), -9219757)
