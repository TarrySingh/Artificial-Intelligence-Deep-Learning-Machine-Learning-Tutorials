'''PyTorch-compatible datasets.

Guaranteed to implement `__len__`, and `__getitem__`.

See: http://pytorch.org/docs/0.3.1/data.html
'''

import torch
from PIL import Image
import torch.utils.data

from robosat.tiles import tiles_from_slippy_map, buffer_tile_image


# Single Slippy Map directory structure
class SlippyMapTiles(torch.utils.data.Dataset):
    '''Dataset for images stored in slippy map format.
    '''

    def __init__(self, root, transform=None):
        super().__init__()

        self.tiles = []
        self.transform = transform

        self.tiles = [(tile, path) for tile, path in tiles_from_slippy_map(root)]
        self.tiles.sort(key=lambda tile: tile[0])

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, i):
        tile, path = self.tiles[i]
        image = Image.open(path)

        if self.transform is not None:
            image = self.transform(image)

        return image, tile


# Multiple Slippy Map directories.
# Think: one with images, one with masks, one with rasterized traces.
class SlippyMapTilesConcatenation(torch.utils.data.Dataset):
    '''Dataset to concate multiple input images stored in slippy map format.
    '''

    def __init__(self, inputs, input_transforms, target, target_transform):
        super().__init__()

        # No-op transform needs to be expressed with identify function `id`
        assert len(inputs) == len(input_transforms), 'one transform per input directory'
        assert len(inputs) > 0, 'at least one input slippy map dataset to compose'

        self.inputs = [SlippyMapTiles(inp, fn) for inp, fn in zip(inputs, input_transforms)]
        self.target = SlippyMapTiles(target, target_transform)

        assert len(set([len(dataset) for dataset in self.inputs])) == 1, 'same number of tiles in all inputs'
        assert len(self.target) == len(self.inputs[0]), 'same number of tiles in inputs and target'

    def __len__(self):
        return len(self.target)

    def __getitem__(self, i):
        # at this point all transformations are applied and we expect to work with raw tensors
        inputs = [dataset[i] for dataset in self.inputs]

        images = [image for image, _ in inputs]
        tiles = [tile for _, tile in inputs]

        mask, mask_tile = self.target[i]

        assert len(set(tiles)) == 1, 'all images are for the same tile'
        assert tiles[0] == mask_tile, 'image tile is the same as mask tile'

        return torch.cat(images, dim=0), mask, tiles


# Todo: once we have the SlippyMapDataset this dataset should wrap
# it adding buffer and unbuffer glue on top of the raw tile dataset.
class BufferedSlippyMapDirectory(torch.utils.data.Dataset):
    '''Dataset for buffered slippy map tiles with overlap.
    '''

    def __init__(self, root, transform=None, size=512, overlap=32):
        '''
        Args:
          root: the slippy map directory root with a `z/x/y.png` sub-structure.
          transform: the transformation to run on the buffered tile.
          size: the Slippy Map tile size in pixels
          overlap: the tile border to add on every side; in pixel.

        Note:
          The overlap must not span multiple tiles.

          Use `unbuffer` to get back the original tile.
        '''

        super().__init__()

        assert overlap >= 0
        assert size >= 256

        self.transform = transform
        self.size = size
        self.overlap = overlap
        self.tiles = list(tiles_from_slippy_map(root))

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, i):
        tile, path = self.tiles[i]
        image = buffer_tile_image(tile, self.tiles, overlap=self.overlap, tile_size=self.size)

        if self.transform is not None:
            image = self.transform(image)

        return image, torch.IntTensor([tile.x, tile.y, tile.z])

    def unbuffer(self, probs):
        '''Removes borders from segmentation probabilities added to the original tile image.

        Args:
          probs: the segmentation probability mask to remove buffered borders.

        Returns:
          The probability mask with the original tile's dimensions without added overlap borders.
        '''

        o = self.overlap
        _, x, y = probs.shape

        return probs[:, o : x - o, o: y - o]
