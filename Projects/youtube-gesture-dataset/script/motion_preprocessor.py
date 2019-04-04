# ------------------------------------------------------------------------------
# Copyright (c) ETRI. All rights reserved.
# Licensed under the BSD 3-Clause License.
# This file is part of Youtube-Gesture-Dataset, a sub-project of AIR(AI for Robots) project.
# You can refer to details of AIR project at https://aiforrobots.github.io
# Written by Youngwoo Yoon (youngwoo@etri.re.kr)
# ------------------------------------------------------------------------------

from scipy.signal import savgol_filter
import numpy as np
from scipy.stats import circvar


def normalize_skeleton(data, resize_factor=None):
    def distance(x1, y1, x2, y2):
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    anchor_pt = (data[1 * 2], data[1 * 2 + 1])  # neck
    if resize_factor is None:
        neck_height = float(abs(data[1] - data[1 * 2 + 1]))
        shoulder_length = distance(data[1 * 2], data[1 * 2 + 1], data[2 * 2], data[2 * 2 + 1]) + \
                          distance(data[1 * 2], data[1 * 2 + 1], data[5 * 2], data[5 * 2 + 1])
        resized_neck_height = neck_height / float(shoulder_length)
        if resized_neck_height > 0.6:
            resize_factor = shoulder_length * resized_neck_height / 0.6
        else:
            resize_factor = shoulder_length

    normalized_data = data.copy()
    for i in range(0, len(data), 2):
        normalized_data[i] = (data[i] - anchor_pt[0]) / resize_factor
        normalized_data[i + 1] = (data[i + 1] - anchor_pt[1]) / resize_factor

    return normalized_data, resize_factor


class MotionPreprocessor:
    def __init__(self, skeletons):
        self.skeletons = np.array(skeletons)
        self.filtering_message = "PASS"

    def get(self):
        assert (self.skeletons is not None)

        # filtering
        if self.has_missing_frames():
            self.skeletons = []
            self.filtering_message = "too many missing frames"

        # fill missing joints
        if self.skeletons != []:
            self.fill_missing_joints()
            if self.skeletons is None or np.isnan(self.skeletons).any():
                self.filtering_message = "failed to fill missing joints"
                self.skeletons = []

        # filtering
        if self.skeletons != []:
            if self.is_static():
                self.skeletons = []
                self.filtering_message = "static motion"
            elif self.has_jumping_joint():
                self.skeletons = []
                self.filtering_message = "jumping joint"

        # preprocessing
        if self.skeletons != []:

            self.smooth_motion()

            is_side_view = False
            self.skeletons = self.skeletons.tolist()
            for i, frame in enumerate(self.skeletons):
                del frame[2::3]  # remove confidence values
                self.skeletons[i], _ = normalize_skeleton(frame)  # translate and scale

                # assertion: missing joints
                assert not np.isnan(self.skeletons[i]).any()

                # side view check
                if (self.skeletons[i][0] < min(self.skeletons[i][2 * 2],
                                               self.skeletons[i][5 * 2]) or
                    self.skeletons[i][0] > max(self.skeletons[i][2 * 2],
                                               self.skeletons[i][5 * 2])):
                    is_side_view = True
                    break

            if len(self.skeletons) == 0 or is_side_view:
                self.filtering_message = "sideview"
                self.skeletons = []

        return self.skeletons, self.filtering_message

    def is_static(self, verbose=False):
        def joint_angle(p1, p2, p3):
            v1 = p1 - p2
            v2 = p3 - p2
            ang1 = np.arctan2(*v1[::-1])
            ang2 = np.arctan2(*v2[::-1])
            return np.rad2deg((ang1 - ang2) % (2 * np.pi))

        def get_joint_variance(skeleton, index1, index2, index3):
            angles = []

            for i in range(skeleton.shape[0]):
                x1, y1 = skeleton[i, index1 * 3], skeleton[i, index1 * 3 + 1]
                x2, y2 = skeleton[i, index2 * 3], skeleton[i, index2 * 3 + 1]
                x3, y3 = skeleton[i, index3 * 3], skeleton[i, index3 * 3 + 1]
                angle = joint_angle(np.array([x1, y1]), np.array([x2, y2]), np.array([x3, y3]))
                angles.append(angle)

            variance = circvar(angles, low=0, high=360)
            return variance

        left_arm_var = get_joint_variance(self.skeletons, 2, 3, 4)
        right_arm_var = get_joint_variance(self.skeletons, 5, 6, 7)

        th = 150
        if left_arm_var < th and right_arm_var < th:
            print('too static - left var {}, right var {}'.format(left_arm_var, right_arm_var))
            return True
        else:
            if verbose:
                print('not static - left var {}, right var {}'.format(left_arm_var, right_arm_var))
            return False

    def has_jumping_joint(self, verbose=False):
        frame_diff = np.squeeze(self.skeletons[1:, :24] - self.skeletons[:-1, :24])
        diffs = abs(frame_diff.flatten())
        width = max(self.skeletons[0, :24:3]) - min(self.skeletons[0, :24:3])

        if max(diffs) > width / 2.0:
            print('jumping joint - diff {}, width {}'.format(max(diffs), width))
            return True
        else:
            if verbose:
                print('no jumping joint - diff {}, width {}'.format(max(diffs), width))
            return False

    def has_missing_frames(self):
        n_empty_frames = 0
        n_frames = self.skeletons.shape[0]
        for i in range(n_frames):
            if np.sum(self.skeletons[i]) == 0:
                n_empty_frames += 1

        ret = n_empty_frames > n_frames * 0.1
        if ret:
            print('missing frames - {} / {}'.format(n_empty_frames, n_frames))
        return ret

    def smooth_motion(self):
        for i in range(24):
            self.skeletons[:, i] = savgol_filter(self.skeletons[:, i], 5, 2)

    def fill_missing_joints(self):
        skeletons = self.skeletons
        n_joints = 8  # only upper body

        def nan_helper(y):
            return np.isnan(y), lambda z: z.nonzero()[0]

        for i in range(n_joints):
            xs, ys = skeletons[:, i * 3], skeletons[:, i * 3 + 1]
            xs[xs == 0] = np.nan
            ys[ys == 0] = np.nan

            if sum(np.isnan(xs)) > len(xs) / 2:
                skeletons = None
                break

            if sum(np.isnan(ys)) > len(ys) / 2:
                skeletons = None
                break

            if np.isnan(xs).any():
                nans, t = nan_helper(xs)
                xs[nans] = np.interp(t(nans), t(~nans), xs[~nans])
                skeletons[:, i * 3] = xs

            if np.isnan(ys).any():
                nans, t = nan_helper(ys)
                ys[nans] = np.interp(t(nans), t(~nans), ys[~nans])
                skeletons[:, i * 3 + 1] = ys

        return skeletons
