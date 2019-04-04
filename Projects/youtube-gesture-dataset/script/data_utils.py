# ------------------------------------------------------------------------------
# Copyright (c) ETRI. All rights reserved.
# Licensed under the BSD 3-Clause License.
# This file is part of Youtube-Gesture-Dataset, a sub-project of AIR(AI for Robots) project.
# You can refer to details of AIR project at https://aiforrobots.github.io
# Written by Youngwoo Yoon (youngwoo@etri.re.kr)
# ------------------------------------------------------------------------------

import glob
import matplotlib
import cv2
import re
import json
import _pickle as pickle
from webvtt import WebVTT
from config import my_config


###############################################################################
# SKELETON
def draw_skeleton_on_image(img, skeleton, thickness=15):
    if not skeleton:
        return img

    new_img = img.copy()
    for pair in SkeletonWrapper.skeleton_line_pairs:
        pt1 = (int(skeleton[pair[0] * 3]), int(skeleton[pair[0] * 3 + 1]))
        pt2 = (int(skeleton[pair[1] * 3]), int(skeleton[pair[1] * 3 + 1]))
        if pt1[0] == 0 or pt2[1] == 0:
            pass
        else:
            rgb = [v * 255 for v in matplotlib.colors.to_rgba(pair[2])][:3]
            cv2.line(new_img, pt1, pt2, color=rgb[::-1], thickness=thickness)

    return new_img


def is_list_empty(my_list):
    return all(map(is_list_empty, my_list)) if isinstance(my_list, list) else False


def get_closest_skeleton(frame, selected_body):
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


def get_skeleton_from_frame(frame):
    if 'pose_keypoints_2d' in frame:
        return frame['pose_keypoints_2d']
    elif 'pose_keypoints' in frame:
        return frame['pose_keypoints']
    else:
        return None


class SkeletonWrapper:
    # color names: https://matplotlib.org/mpl_examples/color/named_colors.png
    visualization_line_pairs = [(0, 1, 'b'), (1, 2, 'darkred'), (2, 3, 'r'), (3, 4, 'gold'), (1, 5, 'darkgreen'), (5, 6, 'g'),
                                (6, 7, 'lightgreen'),
                                (1, 8, 'darkcyan'), (8, 9, 'c'), (9, 10, 'skyblue'), (1, 11, 'deeppink'), (11, 12, 'hotpink'), (12, 13, 'lightpink')]
    skeletons = []
    skeleton_line_pairs = [(0, 1, 'b'), (1, 2, 'darkred'), (2, 3, 'r'), (3, 4, 'gold'), (1, 5, 'darkgreen'),
                           (5, 6, 'g'), (6, 7, 'lightgreen')]

    def __init__(self, basepath, vid):
        # load skeleton data (and save it to pickle for next load)
        pickle_file = glob.glob(basepath + '/' + vid + '.pickle')

        if pickle_file:
            with open(pickle_file[0], 'rb') as file:
                self.skeletons = pickle.load(file)
        else:
            files = glob.glob(basepath + '/' + vid + '/*.json')
            if len(files) > 10:
                files = sorted(files)
                self.skeletons = []
                for file in files:
                    self.skeletons.append(self.read_skeleton_json(file))
                with open(basepath + '/' + vid + '.pickle', 'wb') as file:
                    pickle.dump(self.skeletons, file)
            else:
                self.skeletons = []


    def read_skeleton_json(self, file):
        with open(file) as json_file:
            skeleton_json = json.load(json_file)
            return skeleton_json['people']


    def get(self, start_frame_no, end_frame_no, interval=1):

        chunk = self.skeletons[start_frame_no:end_frame_no]

        if is_list_empty(chunk):
            return []
        else: 
            if interval > 1:
                return chunk[::int(interval)]
            else:
                return chunk


###############################################################################
# VIDEO
def read_video(base_path, vid):
    files = glob.glob(base_path + '/*' + vid + '.mp4')
    if len(files) == 0:
        return None
    elif len(files) >= 2:
        assert False
    filepath = files[0]

    video_obj = VideoWrapper(filepath)

    return video_obj


class VideoWrapper:
    video = []

    def __init__(self, filepath):
        self.filepath = filepath
        self.video = cv2.VideoCapture(filepath)
        self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.height = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.framerate = self.video.get(cv2.CAP_PROP_FPS)

    def get_video_reader(self):
        return self.video

    def frame2second(self, frame_no):
        return frame_no / self.framerate

    def second2frame(self, second):
        return int(round(second * self.framerate))

    def set_current_frame(self, cur_frame_no):
        self.video.set(cv2.CAP_PROP_POS_FRAMES, cur_frame_no)


###############################################################################
# CLIP
def load_clip_data(vid):
    try:
        with open("{}/{}.json".format(my_config.CLIP_PATH, vid)) as data_file:
            data = json.load(data_file)
            return data
    except FileNotFoundError:
        return None


def load_clip_filtering_aux_info(vid):
    try:
        with open("{}/{}_aux_info.json".format(my_config.CLIP_PATH, vid)) as data_file:
            data = json.load(data_file)
            return data
    except FileNotFoundError:
        return None


#################################################################################
#SUBTITLE
class SubtitleWrapper:
    TIMESTAMP_PATTERN = re.compile('(\d+)?:?(\d{2}):(\d{2})[.,](\d{3})')

    def __init__(self, vid, mode):
        self.subtitle = []
        if mode == 'auto':
            self.load_auto_subtitle_data(vid)
        elif mode == 'gentle':
            self.laod_gentle_subtitle(vid)

    def get(self):
        return self.subtitle

    # using gentle lib
    def laod_gentle_subtitle(self,vid):
        try:
            with open("{}/{}_align_results.json".format(my_config.VIDEO_PATH, vid)) as data_file:
                data = json.load(data_file)
                if 'words' in data:
                    raw_subtitle = data['words']

                    for word in raw_subtitle :
                        if word['case'] == 'success':
                            self.subtitle.append(word)
                else:
                    self.subtitle = None
                return data
        except FileNotFoundError:
            self.subtitle = None

    # using youtube automatic subtitle
    def load_auto_subtitle_data(self, vid):
        lang = my_config.LANG
        postfix_in_filename = '-'+lang+'-auto.vtt'
        file_list = glob.glob(my_config.SUBTITLE_PATH + '/*' + vid + postfix_in_filename)
        if len(file_list) > 1:
            print('more than one subtitle. check this.', file_list)
            self.subtitle = None
            assert False
        if len(file_list) == 1:
            for i, subtitle_chunk in enumerate(WebVTT().read(file_list[0])):
                raw_subtitle = str(subtitle_chunk.raw_text)
                if raw_subtitle.find('\n'):
                    raw_subtitle = raw_subtitle.split('\n')

                for raw_subtitle_chunk in raw_subtitle:
                    if self.TIMESTAMP_PATTERN.search(raw_subtitle_chunk) is None:
                        continue

                    # removes html tags and timing tags from caption text
                    raw_subtitle_chunk = raw_subtitle_chunk.replace("</c>", "")
                    raw_subtitle_chunk = re.sub("<c[.]\w+>", '', raw_subtitle_chunk)

                    word_list = []
                    raw_subtitle_s = subtitle_chunk.start_in_seconds
                    raw_subtitle_e = subtitle_chunk.end_in_seconds

                    word_chunk = raw_subtitle_chunk.split('<c>')

                    for i, word in enumerate(word_chunk):
                        word_info = {}

                        if i == len(word_chunk)-1:
                            word_info['word'] = word
                            word_info['start'] = word_list[i-1]['end']
                            word_info['end'] = raw_subtitle_e
                            word_list.append(word_info)
                            break

                        word = word.split("<")
                        word_info['word'] = word[0]
                        word_info['end'] = self.get_seconds(word[1][:-1])

                        if i == 0:
                            word_info['start'] = raw_subtitle_s
                            word_list.append(word_info)
                            continue

                        word_info['start'] = word_list[i-1]['end']
                        word_list.append(word_info)

                    self.subtitle.extend(word_list)
        else:
            print('subtitle file is not exist')
            self.subtitle = None

    # convert timestamp to second
    def get_seconds(self, word_time_e):
        time_value = re.match(self.TIMESTAMP_PATTERN, word_time_e)
        if not time_value:
            print('wrong time stamp pattern')
            exit()

        values = list(map(lambda x: int(x) if x else 0, time_value.groups()))
        hours, minutes, seconds, milliseconds = values[0], values[1], values[2], values[3]

        return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
