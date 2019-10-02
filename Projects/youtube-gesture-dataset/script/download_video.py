# ------------------------------------------------------------------------------
# Copyright (c) ETRI. All rights reserved.
# Licensed under the BSD 3-Clause License.
# This file is part of Youtube-Gesture-Dataset, a sub-project of AIR(AI for Robots) project.
# You can refer to details of AIR project at https://aiforrobots.github.io
# Written by Youngwoo Yoon (youngwoo@etri.re.kr)
# ------------------------------------------------------------------------------

from __future__ import unicode_literals

import glob
import json
import traceback

import youtube_dl
import urllib.request
import sys
import os
from apiclient.discovery import build
from datetime import datetime, timedelta
from config import my_config

YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

RESUME_VIDEO_ID = ""  # resume downloading from this video, set empty string to start over


def fetch_video_ids(channel_id, search_start_time):  # load video ids in the channel
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=my_config.DEVELOPER_KEY)

    start_time = search_start_time
    td = timedelta(days=15)
    end_time = start_time + td

    res_items = []

    # multiple quires are necessary to get all results surely
    while start_time < datetime.now():
        start_string = str(start_time.isoformat()) + 'Z'
        end_string = str(end_time.isoformat()) + 'Z'

        res = youtube.search().list(part="id", channelId=channel_id, maxResults="50",
                                    publishedAfter=start_string,
                                    publishedBefore=end_string).execute()
        res_items += res['items']

        while True:  # paging
            if len(res['items']) < 50 or 'nextPageToken' not in res:
                break

            next_page_token = res['nextPageToken']
            res = youtube.search().list(part="id", channelId=channel_id, maxResults="50",
                                        publishedAfter=start_string,
                                        publishedBefore=end_string,
                                        pageToken=next_page_token).execute()
            res_items += res['items']

        print('    {} to {}, no of videos: {}'.format(start_string, end_string, len(res_items)))

        start_time = end_time
        end_time = start_time + td

    # collect video ids
    vid_list = []
    for i in res_items:
        vid = (i.get('id')).get('videoId')
        if vid is not None:
            vid_list.append(vid)

    return vid_list


def video_filter(info):
    passed = True

    exist_proper_format = False
    format_data = info.get('formats')
    for i in format_data:
        if i.get('ext') == 'mp4' and i.get('height') >= 720 and i.get('acodec') != 'none':
            exist_proper_format = True
    if not exist_proper_format:
        passed = False

    if passed:
        duration_hours = info.get('duration') / 3600.0
        if duration_hours > 1.0:
            passed = False

    if passed:
        if len(info.get('automatic_captions')) == 0 and len(info.get('subtitles')) == 0:
            passed = False

    return passed


def download_subtitle(url, filename, postfix):
    urllib.request.urlretrieve(url, '{}-{}.vtt'.format(filename, postfix))


def download(vid_list):
    ydl_opts = {'format': 'best[height=720,ext=mp4]',
                'writesubtitles': True,
                'writeautomaticsub': True,
                'outtmpl': 'dummy.mp4'
                }  # download options
    language = my_config.LANG

    download_count = 0
    skip_count = 0
    sub_count = 0
    log = open("download_log.txt", 'w', encoding="utf-8")

    if len(RESUME_VIDEO_ID) < 10:
        skip_index = 0
    else:
        skip_index = vid_list.index(RESUME_VIDEO_ID)

    for i in range(len(vid_list)):
        error_count = 0
        print(vid_list[i])
        if i < skip_index:
            continue

        # rename video (vid.mp4)
        ydl_opts['outtmpl'] = my_config.VIDEO_PATH + '/' + vid_list[i] + '.mp4'

        # check existing file
        if os.path.exists(ydl_opts['outtmpl']) and os.path.getsize(ydl_opts['outtmpl']):  # existing and not empty
            print('video file already exists ({})'.format(vid_list[i]))
            continue

        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            vid = vid_list[i]
            url = "https://youtu.be/{}".format(vid)

            info = ydl.extract_info(url, download=False)
            if video_filter(info):
                with open("{}.json".format(vid), "w", encoding="utf-8") as js:
                    json.dump(info, js)
                while 1:
                    if error_count == 3:
                        print('Exit...')
                        sys.exit()
                    try:
                        ydl.download([url])
                    except(youtube_dl.utils.DownloadError,
                           youtube_dl.utils.ContentTooShortError,
                           youtube_dl.utils.ExtractorError):
                        error_count += 1
                        print('  Retrying... (error count : {})\n'.format(error_count))
                        traceback.print_exc()
                        continue
                    else:
                        if info.get('subtitles') != {} and (info.get('subtitles')).get(language) != None:
                            sub_url = (((info.get('subtitles')).get(language))[1]).get('url')
                            download_subtitle(sub_url, vid, language)
                            sub_count += 1
                        if info.get('automatic_captions') != {}:
                            auto_sub_url = (((info.get('automatic_captions')).get(language))[1]).get('url')
                            download_subtitle(auto_sub_url, vid, language+'-auto')

                        log.write("{} - downloaded\n".format(str(vid)))
                        download_count += 1
                        break
            else:
                log.write("{} - skipped\n".format(str(info.get('id'))))
                skip_count += 1

        print("  downloaded: {}, skipped: {}".format(download_count, skip_count))

    log.write("\nno of subtitles : {}\n".format(sub_count))
    log.write("downloaded: {}, skipped : {}\n".format(download_count, skip_count))
    log.close()


def main():
    if not os.path.exists(my_config.VIDEO_PATH):
        os.makedirs(my_config.VIDEO_PATH)

    os.chdir(my_config.VIDEO_PATH)
    vid_list = []

    # read video list
    try:
        rf = open("video_ids.txt", 'r')
    except FileNotFoundError:
        print("fetching video ids...")
        vid_list = fetch_video_ids(my_config.YOUTUBE_CHANNEL_ID, my_config.VIDEO_SEARCH_START_DATE)
        wf = open("video_ids.txt", "w")
        for j in vid_list:
            wf.write(str(j))
            wf.write('\n')
        wf.close()
    else:
        while 1:
            value = rf.readline()[:11]
            if value == '':
                break
            vid_list.append(value)
        rf.close()

    print("downloading videos...")
    download(vid_list)
    print("finished downloading videos")

    print("removing unnecessary subtitles...")
    for f in glob.glob("*.en.vtt"):
        os.remove(f)


def test_fetch():
    vid_list = fetch_video_ids(my_config.YOUTUBE_CHANNEL_ID, my_config.VIDEO_SEARCH_START_DATE)
    print(vid_list)
    print(len(vid_list))


if __name__ == '__main__':
    # test_fetch()
    main()

