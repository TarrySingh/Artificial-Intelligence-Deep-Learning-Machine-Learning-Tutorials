'''Slippy Map Tiles.

The Slippy Map tile spec works with a directory structure of `z/x/y.png` where
- `z` is the zoom level
- `x` is the left / right index
- `y` is the top / bottom index

See: https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames
'''

import csv
import io
import os

from PIL import Image
import mercantile


def pixel_to_location(tile, dx, dy):
    '''Converts a pixel in a tile to a coordinate.

    Args:
      tile: the mercantile tile to calculate the location in.
      dx: the relative x offset in range [0, 1].
      dy: the relative y offset in range [0, 1].

    Returns:
      The coordinate for the pixel in the tile.
    '''

    assert 0 <= dx <= 1, 'x offset is in [0, 1]'
    assert 0 <= dy <= 1, 'y offset is in [0, 1]'

    west, south, east, north = mercantile.bounds(tile)

    def lerp(a, b, c):
        return a + c * (b - a)

    lon = lerp(west, east, dx)
    lat = lerp(south, north, dy)

    return lon, lat


def fetch_image(session, url, timeout=10):
    '''Fetches the image representation for a tile.

    Args:
      session: the HTTP session to fetch the image from.
      url: the tile imagery's url to fetch the image from.
      timeout: the HTTP timeout in seconds.

    Returns:
     The satellite imagery as bytes or None in case of error.
    '''

    try:
        resp = session.get(url, timeout=timeout)
        resp.raise_for_status()
        return io.BytesIO(resp.content)
    except Exception:
        return None


def tiles_from_slippy_map(root):
    '''Loads files from an on-disk slippy map directory structure.

    Args:
      root: the base directory with layout `z/x/y.*`.

    Yields:
      The mercantile tiles and file paths from the slippy map directory.
    '''

    for z in os.listdir(root):
        for x in os.listdir(os.path.join(root, z)):
            for name in os.listdir(os.path.join(root, z, x)):
                y = os.path.splitext(name)[0]
                tile = mercantile.Tile(x=int(x), y=int(y), z=int(z))
                path = os.path.join(root, z, x, name)
                yield tile, path


def tiles_from_csv(path):
    '''Read tiles from a line-delimited csv file.

    Args:
      file: the path to read the csv file from.

    Yields:
      The mercantile tiles from the csv file.
    '''

    with open(path) as fp:
        reader = csv.reader(fp)

        for row in reader:
            if not row:
                continue

            yield mercantile.Tile(*map(int, row))


def stitch_image(into, into_box, image, image_box):
    '''Stitches two images together in-place.

    Args:
      into: the image to stitch into and modify in-place.
      into_box: left, upper, right, lower image coordinates for where to place `image` in `into`.
      image: the image to stitch into `into`.
      image_box: left, upper, right, lower image coordinates for where to extract the sub-image from `image`.

    Note:
      Both boxes must be of same size.
    '''

    into.paste(image.crop(box=image_box), box=into_box)


def adjacent_tile(tile, dx, dy, tiles):
    '''Retrieves an adjacent tile from a tile store.

    Args:
      tile: the original tile to get an adjacent tile for.
      dx: the offset in tile x direction.
      dy: the offset in tile y direction.
      tiles: the tile store to get tiles from; must support `__getitem__` with tiles.

    Returns:
      The adjacent tile's image or `None` if it does not exist.
    '''

    x, y, z = map(int, [tile.x, tile.y, tile.z])
    other = mercantile.Tile(x=x + dx, y=y + dy, z=z)

    try:
        path = tiles[other]
        return Image.open(path).convert('RGB')
    except KeyError:
        return None


def buffer_tile_image(tile, tiles, overlap, tile_size, nodata=0):
    '''Buffers a tile image adding borders on all sides based on adjacent tiles.

    Args:
      tile: the tile to buffer.
      tiles: available tiles; must be a mapping of tiles to their filesystem paths.
      overlap: the tile border to add on every side; in pixel.
      tile_size: the tile size.
      nodata: the color value to use when no adjacent tile is available.

    Returns:
      The composite image containing the original tile plus tile overlap on all sides.
      It's size is `tile_size` + 2 * `overlap` pixel for each side.
    '''

    tiles = dict(tiles)
    x, y, z = map(int, [tile.x, tile.y, tile.z])

    # Todo: instead of nodata we should probably mirror the center image
    composite_size = tile_size + 2 * overlap
    composite = Image.new(mode='RGB', size=(composite_size, composite_size), color=nodata)

    path = tiles[tile]
    center = Image.open(path).convert('RGB')
    composite.paste(center, box=(overlap, overlap))

    top_left = adjacent_tile(tile, -1, -1, tiles)
    top_right = adjacent_tile(tile, +1, -1, tiles)
    bottom_left = adjacent_tile(tile, -1, +1, tiles)
    bottom_right = adjacent_tile(tile, +1, +1, tiles)

    top = adjacent_tile(tile, 0, -1, tiles)
    left = adjacent_tile(tile, -1, 0, tiles)
    bottom = adjacent_tile(tile, 0, +1, tiles)
    right = adjacent_tile(tile, +1, 0, tiles)

    def maybe_stitch(maybe_tile, composite_box, tile_box):
        if maybe_tile:
            stitch_image(composite, composite_box, maybe_tile, tile_box)

    maybe_stitch(top_left, (0, 0, overlap, overlap),
                           (tile_size - overlap, tile_size - overlap, tile_size, tile_size))
    maybe_stitch(top_right, (tile_size + overlap, 0, composite_size, overlap),
                            (0, tile_size - overlap, overlap, tile_size))
    maybe_stitch(bottom_left, (0, composite_size - overlap, overlap, composite_size),
                              (tile_size - overlap, 0, tile_size, overlap))
    maybe_stitch(bottom_right, (composite_size - overlap, composite_size - overlap, composite_size, composite_size),
                               (0, 0, overlap, overlap))
    maybe_stitch(top, (overlap, 0, composite_size - overlap, overlap),
                      (0, tile_size - overlap, tile_size, tile_size))
    maybe_stitch(left, (0, overlap, overlap, composite_size - overlap),
                       (tile_size - overlap, 0, tile_size, tile_size))
    maybe_stitch(bottom, (overlap, composite_size - overlap, composite_size - overlap, composite_size),
                         (0, 0, tile_size, overlap))
    maybe_stitch(right, (composite_size - overlap, overlap, composite_size, composite_size - overlap),
                        (0, 0, overlap, tile_size))

    return composite
