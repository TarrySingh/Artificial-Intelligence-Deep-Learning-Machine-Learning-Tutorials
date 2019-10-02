# ------------------------------------------------------------------------------
# Copyright 2019 ETRI. All rights reserved.
# Licensed under the BSD 3-Clause License.
# This file is part of Youtube-Gesture-Dataset, a sub-project of AIR(AI for Robots) project.
# You can refer to details of AIR project at https://aiforrobots.github.io
# Written by Youngwoo Yoon (youngwoo@etri.re.kr)
# ------------------------------------------------------------------------------

import matplotlib
from config import *
import copy
import os
from data_utils import *
from tqdm import *
from config import *
import numpy as np


class MainSpeakerSelector:
    def __init__(self, raw_skeleton_chunk):
        self.main_speaker_skeletons = self.find_main_speaker_skeletons(raw_skeleton_chunk)

    def get(self):
        return self.main_speaker_skeletons

    def find_main_speaker_skeletons(self, raw_skeleton_chunk):
        tracked_skeletons = []
        selected_skeletons = []  # reference skeleton
        for raw_frame in raw_skeleton_chunk:  # frame
            tracked_person = []
            if selected_skeletons == []:
                # select a main speaker
                confidence_list = []
                for person in raw_frame:  # people
                    body = get_skeleton_from_frame(person)
                    mean_confidence = 0
                    n_points = 0

                    # Calculate the average of confidences of each person
                    for i in range(8):  # upper-body only
                        x = body[i * 3]
                        y = body[i * 3 + 1]
                        confidence = body[i * 3 + 2]
                        if x > 0 and y > 0 and confidence > 0:
                            n_points += 1
                            mean_confidence += confidence
                    if n_points > 0:
                        mean_confidence /= n_points
                    else:
                        mean_confidence = 0
                    confidence_list.append(mean_confidence)

                # select main_speaker with the highest average of confidence
                if len(confidence_list) > 0:
                    max_index = confidence_list.index(max(confidence_list))
                    selected_skeletons = get_skeleton_from_frame(raw_frame[max_index])

            if selected_skeletons != []:
                # find the closest one to the selected main_speaker's skeleton
                tracked_person = self.get_closest_skeleton(raw_frame, selected_skeletons)

            # save
            if tracked_person:
                skeleton_data = tracked_person
                selected_skeletons = get_skeleton_from_frame(tracked_person)
            else:
                skeleton_data = {}

            tracked_skeletons.append(skeleton_data)

        return tracked_skeletons

    def get_closest_skeleton(self, frame, selected_body):
        """ find the closest one to the selected skeleton """
        diff_idx = [i * 3 for i in range(8)] + [i * 3 + 1 for i in range(8)]  # upper-body

        min_diff = 10000000
        tracked_person = None
        for person in frame:  # people
            body = get_skeleton_from_frame(person)

            diff = 0
            n_diff = 0
            for i in diff_idx:
                if body[i] > 0 and selected_body[i] > 0:
                    diff += abs(body[i] - selected_body[i])
                    n_diff += 1
            if n_diff > 0:
                diff /= n_diff
            if diff < min_diff:
                min_diff = diff
                tracked_person = person

        base_distance = max(abs(selected_body[0 * 3 + 1] - selected_body[1 * 3 + 1]) * 3,
                            abs(selected_body[2 * 3] - selected_body[5 * 3]) * 2)
        if tracked_person and min_diff > base_distance:  # tracking failed
            tracked_person = None

        return tracked_person
