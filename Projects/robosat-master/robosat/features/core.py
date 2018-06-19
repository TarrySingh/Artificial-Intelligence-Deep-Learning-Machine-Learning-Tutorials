import cv2
import numpy as np
from PIL import Image

from robosat.tiles import pixel_to_location


def visualize(mask, path):
    '''Writes a visual representation `.png` file for a binary mask.

    Args:
      mask: the binary mask to visualize.
      path: the path to save the `.png` image to.
    '''

    out = Image.fromarray(mask, mode='P')
    out.putpalette([0, 0, 0, 255, 255, 255])
    out.save(path)


def contours_to_mask(contours, shape):
    '''Creates a binary mask for contours.

    Args:
      contours: the contours to create a mask for.
      shape: the resulting mask's shape

    Returns:
      The binary mask with rasterized contours.
    '''

    canvas = np.zeros(shape, np.uint8)
    cv2.drawContours(canvas, contours, contourIdx=-1, color=1)
    return canvas


def featurize(tile, polygon, shape):
    '''Transforms polygons in image pixel coordinates into world coordinates.

    Args:
      tile: the tile this polygon is in for coordinate calculation.
      polygon: the polygon to transform from pixel to world coordinates.
      shape: the image's max x and y coordinates.

    Returns:
      The closed polygon transformed into world coordinates.
    '''

    xmax, ymax = shape

    feature = []

    for point in polygon:
        px, py = point[0]
        dx, dy = px / xmax, py / ymax

        feature.append(pixel_to_location(tile, dx, 1. - dy))

    assert feature, 'at least one location in polygon'
    feature.append(feature[0])  # polygons are closed

    return feature


def denoise(mask, eps):
    '''Removes noise from a mask.

    Args:
      mask: the mask to remove noise from.
      eps: the morphological operation's kernel size for noise removal, in pixel.

    Returns:
      The mask after applying denoising.
    '''

    struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (eps, eps))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, struct)


def grow(mask, eps):
    '''Grows a mask to fill in small holes, e.g. to establish connectivity.

    Args:
      mask: the mask to grow.
      eps: the morphological operation's kernel size for growing, in pixel.

    Returns:
      The mask after filling in small holes.
    '''

    struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (eps, eps))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, struct)


def contours(mask):
    '''Extracts contours and the relationship between them from a binary mask.

    Args:
      mask: the binary mask to find contours in.

    Returns:
      The detected contours as a list of points and the contour hierarchy.

    Note: the hierarchy can be used to re-construct polygons with holes as one entity.
    '''

    _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy


# Todo: should work for lines, too, but then needs other epsilon criterion than arc length
def simplify(polygon, eps):
    '''Simplifies a polygon to minimize the polygon's vertices.

    Args:
      polygon: the polygon made up of a list of vertices.
      eps: the approximation accuracy as max. percentage of the arc length, in [0, 1]

    '''

    assert 0 <= eps <= 1, 'approximation accuracy is percentage in [0, 1]'

    epsilon = eps * cv2.arcLength(polygon, closed=True)
    return cv2.approxPolyDP(polygon, epsilon=epsilon, closed=True)


def parents_in_hierarchy(node, tree):
    '''Walks a hierarchy tree upwards from a starting node collecting all nodes on the way.

    Args:
      node: the index for the starting node in the hierarchy.
      tree: the hierarchy tree containing tuples of (next, prev, first child, parent) ids.

    Yields:
      The node ids on the upwards path in the hierarchy tree.
    '''

    def parent(n):
        # next, prev, fst child, parent
        return n[3]

    at = tree[node]
    up = parent(at)

    while up != -1:
        index = up
        at = tree[index]
        up = parent(at)

        assert index != node, 'upward path does not include starting node'

        yield index
