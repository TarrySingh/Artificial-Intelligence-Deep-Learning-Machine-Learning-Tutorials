# ------------------------------------------------------------------------------
# Copyright (c) ETRI. All rights reserved.
# Licensed under the BSD 3-Clause License.
# This file is part of Youtube-Gesture-Dataset, a sub-project of AIR(AI for Robots) project.
# You can refer to details of AIR project at https://aiforrobots.github.io
# Written by Youngwoo Yoon (youngwoo@etri.re.kr)
# ------------------------------------------------------------------------------

import numpy as np
import cv2
import math

from data_utils import get_skeleton_from_frame
from config import my_config


class ClipFilter:
    def __init__(self, video, start_frame_no, end_frame_no, raw_skeleton, main_speaker_skeletons):
        self.skeleton_data = raw_skeleton
        self.main_speaker_skeletons = main_speaker_skeletons
        self.start_frame_no = start_frame_no
        self.end_frame_no = end_frame_no
        self.scene_length = end_frame_no - start_frame_no
        self.video = video
        self.filter_option = my_config.FILTER_OPTION

        # filtering criteria variable
        self.filtering_results = [0, 0, 0, 0, 0, 0, 0]  # too short, many_people, looking_back, joint_missing, looking_sideways, small, picture
        self.message = ''
        self.debugging_info = ['None', 'None', 'None', 'None', 'None']  # looking back, joint missing, looking sideways, small, picture

    def is_skeleton_back(self, ratio):
        n_incorrect_frame = 0

        for ia, skeleton in enumerate(self.main_speaker_skeletons):  # frames
            body = get_skeleton_from_frame(skeleton)
            if body:
                if body[2 * 3] > body[5 * 3]:
                    n_incorrect_frame += 1
            else:
                n_incorrect_frame += 1

        self.debugging_info[0] = round(n_incorrect_frame / self.scene_length, 3)

        return n_incorrect_frame / self.scene_length > ratio

    def is_skeleton_sideways(self, ratio):
        n_incorrect_frame = 0

        for ia, skeleton in enumerate(self.main_speaker_skeletons):  # frames
            body = get_skeleton_from_frame(skeleton)
            if body:
                if (body[0] < min(body[2 * 3], body[5 * 3]) or body[0] > max(body[2 * 3], body[5 * 3])):
                    n_incorrect_frame += 1
            else:
                n_incorrect_frame += 1

        self.debugging_info[2] = round(n_incorrect_frame / self.scene_length, 3)

        return n_incorrect_frame / self.scene_length > ratio

    def is_skeleton_missing(self, ratio):
        n_incorrect_frame = 0

        if self.main_speaker_skeletons == []:
            n_incorrect_frame = self.scene_length
        else:
            for ia, skeleton in enumerate(self.main_speaker_skeletons):  # frames

                body = get_skeleton_from_frame(skeleton)
                if body:
                    point_idx = [0, 1, 2, 3, 4, 5, 6, 7]  # head and arms
                    if any(body[idx * 3] == 0 for idx in point_idx):
                        n_incorrect_frame += 1

                else:
                    n_incorrect_frame += 1

        self.debugging_info[1] = round(n_incorrect_frame / self.scene_length, 3)
        return n_incorrect_frame / self.scene_length > ratio

    def is_skeleton_small(self, ratio):
        n_incorrect_frame = 0

        def distance(x1, y1, x2, y2):
            return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

        for ia, skeleton in enumerate(self.main_speaker_skeletons):  # frames
            body = get_skeleton_from_frame(skeleton)
            if body:
                threshold = self.filter_option['threshold']  # for TED videos in 720p
                if distance(body[2 * 3], body[2 * 3 + 1], body[5 * 3], body[5 * 3 + 1]) < threshold:  # shoulder length
                    n_incorrect_frame += 1
            else:
                n_incorrect_frame += 1

        self.debugging_info[3] = round(n_incorrect_frame / self.scene_length, 3)
        return n_incorrect_frame / self.scene_length > ratio

    def is_too_short(self):
        MIN_SCENE_LENGTH = 25 * 3  # assumed fps = 25
        return self.scene_length < MIN_SCENE_LENGTH

    def is_picture(self):
        sampling_interval = int(math.floor(self.scene_length / 5))
        sampling_frames = list(range(self.start_frame_no + sampling_interval,
                                     self.end_frame_no - sampling_interval + 1, sampling_interval))
        frames = []
        for frame_no in sampling_frames:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = self.video.read()
            frames.append(frame)

        diff = 0
        n_diff = 0
        for frame, next_frame in zip(frames, frames[1:]):
            diff += cv2.norm(frame, next_frame, cv2.NORM_L1)  # abs diff
            n_diff += 1
        diff /= n_diff
        self.debugging_info[4] = round(diff, 0)

        return diff < 3000000

    def is_many_people(self):
        n_people = []
        for skeleton in self.skeleton_data:
            n_people.append(len(skeleton))

        return len(n_people) > 0 and np.mean(n_people) > 5

    def is_correct_clip(self):
        # check if the clip is too short.
        if self.is_too_short():
            self.message = "too Short"
            return False
        self.filtering_results[0] = 1

        # check if there are too many people on the clip
        if self.is_many_people():
            self.message = "too many people"
            return False
        self.filtering_results[1] = 1

        # check if the ratio of back-facing skeletons in the clip exceeds the reference ratio
        if self.is_skeleton_back(0.3):
            self.message = "looking behind"
            return False
        self.filtering_results[2] = 1

        # check if the ratio of skeletons that missing joint in the clip exceeds the reference ratio
        if self.is_skeleton_missing(0.5):
            self.message = "too many missing joints"
            return False
        self.filtering_results[3] = 1

        # check if the ratio of sideways skeletons in the clip exceeds the reference ratio
        if self.is_skeleton_sideways(0.5):
            self.message = "looking sideways"
            return False
        self.filtering_results[4] = 1

        # check if the ratio of the too small skeleton in the clip exceeds the reference ratio
        if self.is_skeleton_small(0.5):
            self.message = "too small."
            return False
        self.filtering_results[5] = 1

        # check if the clip is picture
        if self.is_picture():
            self.message = "still picture"
            return False
        self.filtering_results[6] = 1

        self.message = "PASS"
        return True

    def get_filter_variable(self):
        return self.filtering_results, self.message, self.debugging_info
