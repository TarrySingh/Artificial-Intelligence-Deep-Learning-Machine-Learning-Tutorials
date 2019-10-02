# ------------------------------------------------------------------------------
# Copyright (c) ETRI. All rights reserved.
# Licensed under the BSD 3-Clause License.
# This file is part of Youtube-Gesture-Dataset, a sub-project of AIR(AI for Robots) project.
# You can refer to details of AIR project at https://aiforrobots.github.io
# Written by Youngwoo Yoon (youngwoo@etri.re.kr)
# ------------------------------------------------------------------------------

"""
Extract pose skeletons by using OpenPose library
Need proper LD_LIBRARY_PATH before run this script
Pycharm: In RUN > Edit Configurations, add LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
"""

import glob
import json
import os
import pickle
import subprocess

import shutil

from config import my_config

# maximum accuracy, too slow (~1fps)
#OPENPOSE_OPTION = "--net_resolution -1x736 --scale_number 4 --scale_gap 0.25 --hand --hand_scale_number 6 --hand_scale_range 0.4 --face"
OPENPOSE_OPTION = "--face --hand --number_people_max 3"

OUTPUT_SKELETON_PATH = my_config.WORK_PATH + "/temp_skeleton_raw"
OUTPUT_VIDEO_PATH = my_config.WORK_PATH + "/temp_skeleton_video"

RESUME_VID = ""  # resume from this video
SKIP_EXISTING_SKELETON = True  # skip if the skeleton file is existing


def get_vid_from_filename(filename):
    return filename[-15:-4]


def read_skeleton_json(_file):
    with open(_file) as json_file:
        skeleton_json = json.load(json_file)
        return skeleton_json['people']


def save_skeleton_to_pickle(_vid):
    files = glob.glob(OUTPUT_SKELETON_PATH + '/' + _vid + '/*.json')
    if len(files) > 10:
        files = sorted(files)
        skeletons = []
        for file in files:
            skeletons.append(read_skeleton_json(file))
        with open(my_config.SKELETON_PATH + '/' + _vid + '.pickle', 'wb') as file:
            pickle.dump(skeletons, file)


if __name__ == '__main__':
    if not os.path.exists(my_config.SKELETON_PATH):
        os.makedirs(my_config.SKELETON_PATH)
    if not os.path.exists(OUTPUT_SKELETON_PATH):
        os.makedirs(OUTPUT_SKELETON_PATH)
    if not os.path.exists(OUTPUT_VIDEO_PATH):
        os.makedirs(OUTPUT_VIDEO_PATH)

    os.chdir(my_config.OPENPOSE_BASE_DIR)
    if RESUME_VID == "":
        skip_flag = False
    else:
        skip_flag = True

    video_files = glob.glob(my_config.VIDEO_PATH + "/*.mp4")
    for file in sorted(video_files, key=os.path.getmtime):
        print(file)
        vid = get_vid_from_filename(file)
        print(vid)

        skip_iter = False

        # resume check
        if skip_flag and vid == RESUME_VID:
            skip_flag = False
        skip_iter = skip_flag

        # existing skeleton check
        if SKIP_EXISTING_SKELETON:
            if os.path.exists(my_config.SKELETON_PATH + '/' + vid + '.pickle'):
                print('existing skeleton')
                skip_iter = True

        if not skip_iter:
            # create out dir
            skeleton_dir = OUTPUT_SKELETON_PATH + "/" + vid + "/"
            if os.path.exists(skeleton_dir):
                shutil.rmtree(skeleton_dir)
            else:
                os.makedirs(skeleton_dir)

            # extract skeleton
            command = my_config.OPENPOSE_BIN_PATH + " " + OPENPOSE_OPTION + " --video \"" + file + "\""
            # command += " --write_video " + OUTPUT_VIDEO_PATH + "/" + vid + "_result.avi"  # write result video
            command += " --write_json " + skeleton_dir
            print(command)
            subprocess.call(command, shell=True)

            # save skeletons to a pickle file
            save_skeleton_to_pickle(vid)
