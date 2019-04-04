# Copyright (C) 2019 Nan Wu, Jason Phang, Jungkyu Park, Yiqiu Shen, Zhe Huang, Masha Zorin, 
#   Stanisław Jastrzębski, Thibault Févry, Joe Katsnelson, Eric Kim, Stacey Wolfson, Ujas Parikh, 
#   Sushma Gaddam, Leng Leng Young Lin, Kara Ho, Joshua D. Weinstein, Beatriu Reig, Yiming Gao, 
#   Hildegard Toth, Kristine Pysarenko, Alana Lewin, Jiyon Lee, Krystal Airola, Eralda Mema, 
#   Stephanie Chung, Esther Hwang, Naziya Samreen, S. Gene Kim, Laura Heacock, Linda Moy, 
#   Kyunghyun Cho, Krzysztof J. Geras
#
# This file is part of breast_cancer_classifier.
#
# breast_cancer_classifier is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# breast_cancer_classifier is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with breast_cancer_classifier.  If not, see <http://www.gnu.org/licenses/>.
# ==============================================================================
"""
Defines functions used in search_windows_and_centers.py
"""
import math
import numpy as np
import pandas as pd


def get_images_optimal_window_info(image, com, window_dim_ls, step=1,
                                   tl_br_constraint=None):
    """
    Given an image, windom_dim_ls to search over, and constraints on tl_br
    (e.g. rightmost-pixel constraint), return data about the potential
    optimal windows
    """

    image_result_ls = []
    cumsum = get_image_cumsum(image)
    for window_dim in window_dim_ls:
        image_result_ls.append(get_image_optimal_window_info(
            image, com, window_dim,
            step=step, tl_br_constraint=tl_br_constraint, cumsum=cumsum,
        ))

    return pd.DataFrame(image_result_ls)


def get_image_optimal_window_info(image, com, window_dim,
                                  step=1, tl_br_constraint=None, cumsum=None):
    image_dim = image.shape
    if cumsum is None:
        cumsum = get_image_cumsum(image)
    window_area = np.prod(window_dim)
    tl, br = get_candidate_center_topleft_bottomright(
        com=com, image_dim=image_dim, window_dim=window_dim, step=step)
    if tl_br_constraint:
        tl, br = tl_br_constraint(tl=tl, br=br, image=image, window_dim=window_dim)
    y_grid_axis = np.arange(tl[0], br[0], step)
    x_grid_axis = np.arange(tl[1], br[1], step)
    window_center_ls = get_joint_axes(y_grid_axis, x_grid_axis)

    tl_ls, br_ls = get_candidate_topleft_bottomright(
        image_dim=image_dim,
        window_center=window_center_ls,
        window_dim=window_dim,
    )
    partial_sum = v_get_topleft_bottomright_partialsum(
        cumsum=cumsum,
        topleft=tl_ls,
        bottomright=br_ls,
    )
    if len(partial_sum) == 1:
        best_center = tl
        fraction_of_non_zero_pixels = partial_sum[0] / window_area
    else:
        best_sum = partial_sum.max()
        best_center_ls = window_center_ls[partial_sum == best_sum]
        if len(best_center_ls) == 1:
            best_center = best_center_ls[0]
        else:
            best_indices = best_center_ls - com
            best_idx = np.argmin((best_indices ** 2).sum(1))
            best_offset = best_indices[best_idx]
            best_center = com + best_offset
        fraction_of_non_zero_pixels = best_sum / window_area
    return {
        "window_dim_y": window_dim[0],
        "window_dim_x": window_dim[1],
        "best_center_y": best_center[0],
        "best_center_x": best_center[1],
        "fraction": fraction_of_non_zero_pixels,
    }


def get_image_cumsum(image):
    binary = image > 0
    return get_topleft_bottomright_cumsum(binary)


def get_joint_axes(y_grid_axis, x_grid_axis):
    return np.array(np.meshgrid(
        y_grid_axis, x_grid_axis,
    )).T.reshape(-1, 2)


def get_candidate_center_topleft_bottomright(com, image_dim, window_dim, step):
    """
    br returned is exclusive
    """
    half_dim = window_dim // 2
    rem_window_dim = window_dim - half_dim

    # Find extremes of possible centers, based on window size
    center_tl = half_dim
    center_br = image_dim - rem_window_dim

    #
    tl = com - step * ((com - center_tl) // step)
    br = com + step * ((center_br - com) // step)

    # Default to Center
    if tl[0] >= br[0]:
        tl[0] = br[0] = com[0]
    if tl[1] >= br[1]:
        tl[1] = br[1] = com[1]
    return tl, br + 1


def get_candidate_topleft_bottomright(image_dim, window_center, window_dim):
    """
    Given a desired window_center, compute the feasible (array-indexable)
    topleft and bottomright

    implicitly "zero-pads" if window cannot fit in image
    """
    half_dim = window_dim // 2
    rem_half_dim = window_dim - half_dim
    tl = window_center - half_dim
    offset = (-tl).clip(min=0)
    tl = tl.clip(min=0)
    br = window_center + rem_half_dim + offset
    br = br.clip(max=image_dim)
    return tl, br


def get_topleft_bottomright_cumsum(x):
    return np.cumsum(np.cumsum(x, axis=0), axis=1)


def v_get_topleft_bottomright_partialsum(cumsum, topleft, bottomright):
    """
    bottomright is exclusive, to match slicing indices
    """
    tl_x, tl_y = topleft[:, 0], topleft[:, 1]
    br_x, br_y = bottomright[:, 0], bottomright[:, 1]
    assert np.all(br_x >= tl_x)
    assert np.all(br_y >= tl_y)
    assert np.all(br_x <= cumsum.shape[0])
    assert np.all(br_y <= cumsum.shape[1])

    length = len(topleft)
    topslice = np.zeros(length)
    leftslice = np.zeros(length)
    topleft_slice = np.zeros(length)
    bottomright_slice = np.zeros(length)

    selector = (tl_y > 0) & (br_x > 0)
    topslice[selector] = cumsum[br_x - 1, tl_y - 1][selector]

    selector = (tl_x > 0) & (br_y > 0)
    leftslice[selector] = cumsum[tl_x - 1, br_y - 1][selector]

    selector = (tl_x > 0) & (tl_y > 0)
    topleft_slice[selector] = cumsum[tl_x - 1, tl_y - 1][selector]

    selector = (br_x > 0) & (br_y > 0)
    bottomright_slice[selector] = cumsum[br_x - 1, br_y - 1][selector]

    return bottomright_slice - topslice - leftslice + topleft_slice


def get_rightmost_pixel_constraint(rightmost_x):
    """
    Given a rightmost_x (x-coord of rightmost nonzero pixel),
    return a constraint function that remaps candidate tl/brs
    such that the right-edge = rightmost_x

    (Should reduce 2D search to 1D)
    """

    def _f(tl, br, image, window_dim, rightmost_x_=rightmost_x):
        if tl[1] == br[1]:
            # We have no room to shift the center-X anyway
            return tl, br
        half_dim_x = window_dim[1] // 2
        tl = tl.copy()
        br = br.copy()
        new_x = rightmost_x_ - half_dim_x
        tl[1] = new_x - 1
        br[1] = new_x
        return tl, br

    return _f


def get_bottomrightmost_pixel_constraint(rightmost_x, bottommost_y):
    """
    Given a rightmost_x (x-coord of rightmost nonzero pixel),
    return a constraint function that remaps candidate tl/brs
    such that the right-edge = rightmost_x

    (Should reduce 2D search to 1D)
    """

    def _f(tl, br, image, window_dim,
           bottommost_y_=bottommost_y, rightmost_x_=rightmost_x):

        # Check for empty rows at bottom
        relevant_image_from_right = image[:, -window_dim[1]:]
        non_zero_rows = np.any((relevant_image_from_right != 0), axis=1)
        if non_zero_rows.any():
            last_nonzero_row = np.arange(non_zero_rows.shape[0])[non_zero_rows][-1]
            bottommost_y_ = min(last_nonzero_row, bottommost_y_)

        half_dim = window_dim // 2
        br = np.array([bottommost_y_, rightmost_x_]) - half_dim
        tl = br - 1

        return tl, br

    return _f
