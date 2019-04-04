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
Runs search_windows_and_centers.py and extract_centers.py in the same directory
"""
import argparse
import numpy as np
import os
from itertools import repeat
from multiprocessing import Pool

from src.constants import INPUT_SIZE_DICT
import src.utilities.pickling as pickling
import src.utilities.data_handling as data_handling
import src.utilities.reading_images as reading_images
import src.data_loading.loading as loading
import src.optimal_centers.calc_optimal_centers as calc_optimal_centers


def extract_center(datum, image):
    """
    Compute the optimal center for an image
    """
    image = loading.flip_image(image, datum["full_view"], datum['horizontal_flip'])
    if datum["view"] == "MLO":
        tl_br_constraint = calc_optimal_centers.get_bottomrightmost_pixel_constraint(
            rightmost_x=datum["rightmost_points"][1],
            bottommost_y=datum["bottommost_points"][0],
        )
    elif datum["view"] == "CC":
        tl_br_constraint = calc_optimal_centers.get_rightmost_pixel_constraint(
            rightmost_x=datum["rightmost_points"][1]
        )
    else:
        raise RuntimeError(datum["view"])
    optimal_center = calc_optimal_centers.get_image_optimal_window_info(
        image,
        com=np.array(image.shape) // 2,
        window_dim=np.array(INPUT_SIZE_DICT[datum["full_view"]]),
        tl_br_constraint=tl_br_constraint,
    )
    return optimal_center["best_center_y"], optimal_center["best_center_x"]


def load_and_extract_center(datum, data_prefix):
    """
    Load image and computer optimal center
    """
    full_image_path = os.path.join(data_prefix, datum["short_file_path"] + '.png')
    image = reading_images.read_image_png(full_image_path)
    return datum["short_file_path"], extract_center(datum, image)


def get_optimal_centers(data_list, data_prefix, num_processes=1):
    """
    Compute optimal centers for each image in data list
    """
    pool = Pool(num_processes)
    result = pool.starmap(load_and_extract_center, zip(data_list, repeat(data_prefix)))
    return dict(result)


def main(cropped_exam_list_path, data_prefix, output_exam_list_path, num_processes=1):
    exam_list = pickling.unpickle_from_file(cropped_exam_list_path)
    data_list = data_handling.unpack_exam_into_images(exam_list, cropped=True)
    optimal_centers = get_optimal_centers(
        data_list=data_list,
        data_prefix=data_prefix,
        num_processes=num_processes
    )
    data_handling.add_metadata(exam_list, "best_center", optimal_centers)
    os.makedirs(os.path.dirname(output_exam_list_path), exist_ok=True)
    pickling.pickle_to_file(output_exam_list_path, exam_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute and Extract Optimal Centers')
    parser.add_argument('--cropped-exam-list-path')
    parser.add_argument('--data-prefix')
    parser.add_argument('--output-exam-list-path', required=True)
    parser.add_argument('--num-processes', default=20)
    args = parser.parse_args()

    main(
        cropped_exam_list_path=args.cropped_exam_list_path,
        data_prefix=args.data_prefix,
        output_exam_list_path=args.output_exam_list_path,
        num_processes=int(args.num_processes),
    )
