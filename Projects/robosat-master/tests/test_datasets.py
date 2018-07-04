import unittest

import torch
from torchvision.transforms import ToTensor
import mercantile

from robosat.transforms import MaskToTensor
from robosat.datasets import SlippyMapTiles, SlippyMapTilesConcatenation


class TestSlippyMapTiles(unittest.TestCase):

    images = 'tests/fixtures/images/'

    def test_len(self):
        dataset = SlippyMapTiles(TestSlippyMapTiles.images)
        self.assertEqual(len(dataset), 3)

    def test_getitem(self):
        dataset = SlippyMapTiles(TestSlippyMapTiles.images)
        image, tile = dataset[0]

        assert tile == mercantile.Tile(69105, 105093, 18)
        # Inspired by: https://github.com/python-pillow/Pillow/blob/master/Tests/test_image.py#L37-L38
        self.assertEqual(repr(image)[:45], '<PIL.JpegImagePlugin.JpegImageFile image mode')
        self.assertEqual(image.size, (512, 512))

    def test_getitem_with_transform(self):
        # TODO
        pass


class TestSlippyMapTilesConcatenation(unittest.TestCase):

    def test_len(self):
        inputs = ['tests/fixtures/images/']
        input_transforms = [ToTensor(), ]

        target = 'tests/fixtures/labels/'
        target_transform = MaskToTensor()

        dataset = SlippyMapTilesConcatenation(inputs, input_transforms, target, target_transform)

        self.assertEqual(len(dataset), 3)

    def test_getitem(self):
        inputs = ['tests/fixtures/images/']
        input_transforms = [ToTensor(), ]

        target = 'tests/fixtures/labels/'
        target_transform = MaskToTensor()

        dataset = SlippyMapTilesConcatenation(inputs, input_transforms, target, target_transform)

        images, mask, tiles = dataset[0]
        self.assertEqual(tiles[0], mercantile.Tile(69105, 105093, 18))
        self.assertEqual(type(images), torch.Tensor)
        self.assertEqual(type(mask), torch.Tensor)
