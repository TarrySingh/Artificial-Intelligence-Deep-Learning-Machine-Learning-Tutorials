import unittest

import mercantile

from robosat.tiles import tiles_from_slippy_map, tiles_from_csv


class TestSlippyMapTiles(unittest.TestCase):

    def test_slippy_map_directory(self):
        root = 'tests/fixtures/images'
        tiles = [tile for tile in tiles_from_slippy_map(root)]
        self.assertEqual(len(tiles), 3)

        tile, path = tiles[0]
        self.assertEqual(type(tile), mercantile.Tile)
        self.assertEqual(path, 'tests/fixtures/images/18/69105/105093.jpg')


class TestReadTiles(unittest.TestCase):

    def test_read_tiles(self):
        filename = 'tests/fixtures/tiles.csv'
        tiles = [tile for tile in tiles_from_csv(filename)]

        self.assertEqual(len(tiles), 3)
        self.assertEqual(tiles[0], mercantile.Tile(69623, 104945, 18))
