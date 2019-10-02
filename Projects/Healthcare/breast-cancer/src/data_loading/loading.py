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
import numpy as np
from src.constants import VIEWS, INPUT_SIZE_DICT
import src.data_loading.augmentations as augmentations
from src.utilities.reading_images import read_image_mat, read_image_png


def flip_image(image, view, horizontal_flip):
    """
    If training mode, makes all images face right direction.
    In medical, keeps the original directions unless horizontal_flip is set.
    """
    if horizontal_flip == 'NO':
        if VIEWS.is_right(view):
            image = np.fliplr(image)
    elif horizontal_flip == 'YES':
        if VIEWS.is_left(view):
            image = np.fliplr(image)

    return image


def standard_normalize_single_image(image):
    """
    Standardizes an image in-place 
    """
    image -= np.mean(image)
    image /= np.maximum(np.std(image), 10**(-5))


def load_image(image_path, view, horizontal_flip):
    """
    Loads a png or hdf5 image as floats and flips according to its view.
    """
    if image_path.endswith("png"):
        image = read_image_png(image_path)
    elif image_path.endswith("hdf5"):
        image = read_image_mat(image_path)
    else:
        raise RuntimeError()
    image = image.astype(np.float32)
    image = flip_image(image, view, horizontal_flip)
    return image


def load_heatmaps(benign_heatmap_path, malignant_heatmap_path, view, horizontal_flip):
    """
    Loads two heatmaps as one numpy array
    """
    assert bool(benign_heatmap_path) == bool(malignant_heatmap_path)
    benign_heatmap = load_image(benign_heatmap_path, view, horizontal_flip)
    malignant_heatmap = load_image(malignant_heatmap_path, view, horizontal_flip)
    heatmaps = np.stack([benign_heatmap, malignant_heatmap], axis=2)
    return heatmaps


def load_image_and_heatmaps(image_path, benign_heatmap_path, malignant_heatmap_path, view, horizontal_flip):
    """
    Loads an image and its corresponding heatmaps if required
    """
    image = load_image(image_path, view, horizontal_flip)
    assert bool(benign_heatmap_path) == bool(malignant_heatmap_path)
    if benign_heatmap_path:
        heatmaps = load_heatmaps(benign_heatmap_path, malignant_heatmap_path, view, horizontal_flip)
    else:
        heatmaps = None
    return image, heatmaps


def augment_and_normalize_image(image, auxiliary_image, view, best_center, random_number_generator,
                                augmentation, max_crop_noise, max_crop_size_noise):
    """
    Applies augmentation window with random noise in location and size
    and return normalized cropped image. 
    """
    view_input_size = INPUT_SIZE_DICT[view]
    if augmentation:
        cropped_image, cropped_auxiliary_image = augmentations.random_augmentation_best_center(
            image=image,
            input_size=view_input_size,
            random_number_generator=random_number_generator,
            max_crop_noise=max_crop_noise,
            max_crop_size_noise=max_crop_size_noise,
            auxiliary_image=auxiliary_image,
            best_center=best_center,
            view=view,
        )
    else:
        cropped_image, cropped_auxiliary_image = augmentations.random_augmentation_best_center(
            image=image,
            input_size=view_input_size,
            random_number_generator=random_number_generator,
            max_crop_noise=(0, 0),
            max_crop_size_noise=0,
            auxiliary_image=auxiliary_image,
            best_center=best_center,
            view=view,
        )
    
    # For test time only, normalize a copy of the cropped image
    # in order to avoid changing the value of original image which gets augmented multiple times
    cropped_image = cropped_image.copy()
    standard_normalize_single_image(cropped_image)
    
    return cropped_image, cropped_auxiliary_image
