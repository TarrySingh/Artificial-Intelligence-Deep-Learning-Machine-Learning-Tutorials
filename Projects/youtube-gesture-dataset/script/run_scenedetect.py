# ------------------------------------------------------------------------------
# Copyright 2019 ETRI. All rights reserved.
# Licensed under the BSD 3-Clause License.
# This file is part of Youtube-Gesture-Dataset, a sub-project of AIR(AI for Robots) project.
# You can refer to details of AIR project at https://aiforrobots.github.io
# Written by Youngwoo Yoon (youngwoo@etri.re.kr)
# ------------------------------------------------------------------------------

from __future__ import unicode_literals
import subprocess
import glob
import os
from tqdm import tqdm
from config import my_config


def run_pyscenedetect(file_path, vid):  # using Pyscenedetect
    os.chdir(my_config.VIDEO_PATH)

    cmd = 'scenedetect --input "{}" --output "{}" -d 4 detect-content list-scenes'.format(file_path, my_config.CLIP_PATH)
    print('  ' + cmd)
    subprocess.run(cmd, shell=True, check=True)
    subprocess.run("exit", shell=True, check=True)


def main():
    if not os.path.exists(my_config.CLIP_PATH):
        os.makedirs(my_config.CLIP_PATH)

    videos = glob.glob(my_config.VIDEO_PATH + "/*.mp4")
    n_total = len(videos)
    for i, file_path in tqdm(enumerate(sorted(videos, key=os.path.getmtime))):
        print('{}/{}'.format(i+1, n_total))
        vid = os.path.split(file_path)[1][-15:-4]

        csv_files = glob.glob(my_config.CLIP_PATH + "/{}*.csv".format(vid))
        if len(csv_files) > 0 and os.path.getsize(csv_files[0]):  # existing and not empty
            print('  CSV file already exists ({})'.format(vid))
        else:
            run_pyscenedetect(file_path, vid)


if __name__ == '__main__':
    main()
