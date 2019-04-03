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
import cv2
import numpy as np

from src.constants import VIEWS


def shift_window_inside_image(start, end, image_axis_size, input_axis_size):
    """
    If the window goes outside the bound of the image, then shifts it to fit inside the image.
    """
    if start < 0:
        start = 0
        end = start + input_axis_size
    elif end > image_axis_size:
        end = image_axis_size
        start = end - input_axis_size

    return start, end


def zero_pad_and_align_window(image_axis_size, input_axis_size, max_crop_and_size_noise, bidirectional):
    """
    Adds Zero padding to the image if cropped image is smaller than required window size. 
    """
    pad_width = input_axis_size - image_axis_size + max_crop_and_size_noise * (2 if bidirectional else 1)
    assert (pad_width >= 0)

    if bidirectional:
        pad_front = int(pad_width / 2)
        start = max_crop_and_size_noise
    else:
        start, pad_front = 0, 0

    pad_back = pad_width - pad_front
    end = start + input_axis_size
    
    return start, end, pad_front, pad_back


def simple_resize(image_to_resize, size):
    """
    Resizes image to the required size 
    """
    image_resized = cv2.resize(image_to_resize, (size[1], size[0]), interpolation=cv2.INTER_CUBIC)
    if len(image_to_resize.shape) == 3 and len(image_resized.shape) == 2 and image_to_resize.shape[2] == 1:
        image_resized = np.expand_dims(image_resized, 2)

    return image_resized


def crop_image(image, input_size, borders):
    """
    Crops image to the required size using window location
    """
    cropped_image = image[borders[0]: borders[1], borders[2]: borders[3]]

    if ((borders[1] - borders[0]) != input_size[0]) or ((borders[3] - borders[2]) != input_size[1]):
        cropped_image = simple_resize(cropped_image, input_size)

    return cropped_image


def window_location_at_center_point(input_size, center_y, center_x):
    """
    Calculates window location (top, bottom, left, right) 
    given center point and size of augmentation window
    """
    half_height = input_size[0] // 2
    half_width = input_size[1] // 2
    top = center_y - half_height
    bottom = center_y + input_size[0] - half_height
    left = center_x - half_width
    right = center_x + input_size[1] - half_width
    return top, bottom, left, right


def sample_crop_best_center(image, input_size, random_number_generator, max_crop_noise, max_crop_size_noise,
                            best_center, view):
    """
    Crops using the best center point and ideal window size.
    Pads small images to have enough room for crop noise and size noise.
    Applies crop noise in location of the window borders.
    """

    max_crop_noise = np.array(max_crop_noise)
    crop_noise_multiplier = np.zeros(2, dtype=np.float32)

    if max_crop_noise.any():
        # there is no point in sampling crop_noise_multiplier if it's going to be multiplied by (0, 0)
        crop_noise_multiplier = random_number_generator.uniform(low=-1.0, high=1.0, size=2)

    center_y, center_x = best_center

    # get the window around the center point. The window might be outside of the image.
    top, bottom, left, right = window_location_at_center_point(input_size, center_y, center_x)

    pad_y_top, pad_y_bottom, pad_x_right = 0, 0, 0

    if VIEWS.is_cc(view):
        if image.shape[0] < input_size[0] + (max_crop_noise[0] + max_crop_size_noise) * 2:
            # Image is smaller than window size + noise margin in y direction.
            # CC view: pad at both top and bottom
            top, bottom, pad_y_top, pad_y_bottom = zero_pad_and_align_window(image.shape[0], input_size[0],
                                                                             max_crop_noise[0] + max_crop_size_noise,
                                                                             True)
    elif VIEWS.is_mlo(view):
        if image.shape[0] < input_size[0] + max_crop_noise[0] + max_crop_size_noise:
            # Image is smaller than window size + noise margin in y direction.
            # MLO view: only pad at the bottom
            top, bottom, _, pad_y_bottom = zero_pad_and_align_window(image.shape[0], input_size[0],
                                                                     max_crop_noise[0] + max_crop_size_noise, False)
    else:
        raise KeyError("Unknown view", view)

    if image.shape[1] < input_size[1] + max_crop_noise[1] + max_crop_size_noise:
        # Image is smaller than window size + noise margin in x direction.
        left, right, _, pad_x_right = zero_pad_and_align_window(image.shape[1], input_size[1],
                                                                max_crop_noise[1] + max_crop_size_noise, False)

    # Pad image if necessary by allocating new memory and copying contents over
    if pad_y_top > 0 or pad_y_bottom > 0 or pad_x_right > 0:
        new_zero_array = np.zeros((
            image.shape[0] + pad_y_top + pad_y_bottom,
            image.shape[1] + pad_x_right, image.shape[2]), dtype=image.dtype)
        new_zero_array[pad_y_top: image.shape[0] + pad_y_top, 0: image.shape[1]] = image
        image = new_zero_array

    # if window is drawn outside of image, shift it to be inside the image.
    top, bottom = shift_window_inside_image(top, bottom, image.shape[0], input_size[0])
    left, right = shift_window_inside_image(left, right, image.shape[1], input_size[1])

    if top == 0:
        # there is nowhere to shift upwards, we only apply noise downwards
        crop_noise_multiplier[0] = np.abs(crop_noise_multiplier[0])
    elif bottom == image.shape[0]:
        # there is nowhere to shift down, we only apply noise upwards
        crop_noise_multiplier[0] = -np.abs(crop_noise_multiplier[0])
    # else: we do nothing to the noise multiplier

    if left == 0:
        # there is nowhere to shift left, we only apply noise to move right
        crop_noise_multiplier[1] = np.abs(crop_noise_multiplier[1])
    elif right == image.shape[1]:
        # there is nowhere to shift right, we only apply noise to move left
        crop_noise_multiplier[1] = -np.abs(crop_noise_multiplier[1])
    # else: we do nothing to the noise multiplier

    borders = np.array((top, bottom, left, right), dtype=np.int32)

    # Calculate maximum amount of how much the window can move for cropping noise
    top_margin = top
    bottom_margin = image.shape[0] - bottom
    left_margin = left
    right_margin = image.shape[1] - right

    if crop_noise_multiplier[0] >= 0:
        vertical_margin = bottom_margin
    else:
        vertical_margin = top_margin

    if crop_noise_multiplier[1] >= 0:
        horizontal_margin = right_margin
    else:
        horizontal_margin = left_margin

    if vertical_margin < max_crop_noise[0]:
        max_crop_noise[0] = vertical_margin

    if horizontal_margin < max_crop_noise[1]:
        max_crop_noise[1] = horizontal_margin

    crop_noise = np.round(max_crop_noise * crop_noise_multiplier)
    crop_noise = np.array((crop_noise[0], crop_noise[0], crop_noise[1], crop_noise[1]), dtype=np.int32)
    borders = borders + crop_noise

    # this is to make sure that the cropping window isn't outside of the image
    assert (borders[0] >= 0) and (borders[1] <= image.shape[0]) and (borders[2] >= 0) and (borders[3] <= image.shape[
        1]), "Centre of the crop area is sampled such that the borders are outside of the image. Borders: " + str(
        borders) + ', image shape: ' + str(image.shape)

    # return the padded image and cropping window information
    return image, borders


def sample_crop(image, input_size, borders, random_number_generator, max_crop_size_noise):
    """
    Applies size noise of the window borders.
    """
    size_noise_multiplier = random_number_generator.uniform(low=-1.0, high=1.0, size=4)

    top_margin = borders[0]
    bottom_margin = image.shape[0] - borders[1]
    left_margin = borders[2]
    right_margin = image.shape[1] - borders[3]

    max_crop_size_noise = min(max_crop_size_noise, top_margin, bottom_margin, left_margin, right_margin)

    if input_size[0] >= input_size[1]:
        max_crop_size_vertical_noise = max_crop_size_noise
        max_crop_size_horizontal_noise = np.round(max_crop_size_noise * (input_size[1] / input_size[0]))
    elif input_size[0] < input_size[1]:
        max_crop_size_vertical_noise = np.round(max_crop_size_noise * (input_size[0] / input_size[1]))
        max_crop_size_horizontal_noise = max_crop_size_noise
    else:
        raise RuntimeError()

    max_crop_size_noise = np.array((max_crop_size_vertical_noise, max_crop_size_vertical_noise,
                                    max_crop_size_horizontal_noise, max_crop_size_horizontal_noise),
                                   dtype=np.int32)
    size_noise = np.round(max_crop_size_noise * size_noise_multiplier)
    size_noise = np.array(size_noise, dtype=np.int32)
    borders = borders + size_noise

    # this is to make sure that the cropping window isn't outside of the image
    assert (borders[0] >= 0) and (borders[1] <= image.shape[0]) and (borders[2] >= 0) and (borders[3] <= image.shape[
        1]), "Center of the crop area is sampled such that the borders are outside of the image. Borders: " + str(
        borders) + ', image shape: ' + str(image.shape)

    # Sanity check. make sure that the top is above the bottom
    assert borders[1] > borders[0], "Bottom above the top. Top: " + str(borders[0]) + ', bottom: ' + str(borders[1])

    # Sanity check. make sure that the left is left to the right
    assert borders[3] > borders[2], "Left on the right. Left: " + str(borders[2]) + ', right: ' + str(borders[3])

    return borders


def random_augmentation_best_center(image, input_size, random_number_generator, max_crop_noise=(0, 0),
                                    max_crop_size_noise=0, auxiliary_image=None,
                                    best_center=None, view=""):
    """
    Crops augmentation window from a given image 
    by applying noise in location and size of the window.
    """
    joint_image = np.expand_dims(image, 2)
    if auxiliary_image is not None:
        joint_image = np.concatenate([joint_image, auxiliary_image], axis=2)

    joint_image, borders = sample_crop_best_center(joint_image, input_size, random_number_generator, max_crop_noise,
                                                   max_crop_size_noise, best_center, view)
    borders = sample_crop(joint_image, input_size, borders, random_number_generator, max_crop_size_noise)
    sampled_joint_image = crop_image(joint_image, input_size, borders)

    if auxiliary_image is None:
        return sampled_joint_image[:, :, 0], None
    else:
        return sampled_joint_image[:, :, 0], sampled_joint_image[:, :, 1:]
