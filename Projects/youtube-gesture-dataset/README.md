# Youtube Gesture Dataset

This repository contains scripts to build *Youtube Gesture Dataset*.
You can download Youtube videos and transcripts, divide the videos into scenes, and extract human poses.
Please see the project page and paper for the details.  
 
[[Project page]](https://sites.google.com/view/youngwoo-yoon/projects/co-speech-gesture-generation) [[Paper]](https://arxiv.org/abs/1810.12541)

If you have any questions or comments, please feel free to contact the original author by email ([youngwoo@etri.re.kr](mailto:youngwoo@etri.re.kr)).

## Environment

The scripts are tested on Ubuntu 16.04 LTS and Python 3.5.2.  
#### Dependencies 
* [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) (v1.4) for pose estimation
* [PySceneDetect](https://pyscenedetect.readthedocs.io/en/latest/) (v0.5) for video scene segmentation
* [OpenCV](https://pypi.org/project/opencv-python/) (v3.4) for video read
  * We uses FFMPEG. Use latest pip version of opencv-python or build OpenCV with FFMPEG.
* [Gentle](https://github.com/lowerquality/gentle) (Jan. 2019 version) for transcript alignment
  * Add an option `-vn` to resample.py as follows:
    ```python
    cmd = [
        FFMPEG,
        '-loglevel', 'panic',
        '-y',
    ] + offset + [
        '-i', infile,
    ] + duration + [
        '-vn',  # ADDED (it blocks video streams, see the ffmpeg option)
        '-ac', '1', '-ar', '8000',
        '-acodec', 'pcm_s16le',
        outfile
    ]
    ``` 

## A step-by-step guide

1. Set config
   * Update paths and youtube developer key in `config.py` (the directories will be created if not exist).
   * Update target channel ID. The scripts are tested for TED and LaughFactory channels.

2. Execute `download_video.py`
   * Download youtube videos, metadata, and subtitles (./videos/*.mp4, *.json, *.vtt).

3. Execute `run_openpose.py`
   * Run [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) to extract body, hand, and face skeletons for all vidoes (./skeleton/*.pickle). 

4. Execute `run_scenedetect.py`
   * Run [PySceneDetect](https://pyscenedetect.readthedocs.io/en/latest/) to divide videos into scene clips (./clip/*.csv).
  
5. Execute `run_gentle.py`
   * Run [Gentle](https://github.com/lowerquality/gentle) for word-level alignments (./videos/*_align_results.json).
   * You should skip this step if you use auto-generated subtitles. This step is necessary for the TED Talks channel. 

6. Execute `run_clip_filtering.py`
   * Remove inappropriate clips.
   * Save clips with body skeletons (./clip/*.json).

7. *(optional)* Execute `review_filtered_clips.py`
   * Review filtering results.

8. Execute `make_ted_dataset.py`
   * Do some post processing and split into train, validation, and test sets (./script/*.pickle).


## Pre-built TED gesture dataset
 
Running whole data collection pipeline is complex and takes several days, so we provide the pre-built dataset for the videos in the TED channel.  

| | |
| --- | --- |
| Number of videos | 1,766 |
| Average length of videos | 12.7 min |
| Shots of interest | 35,685 (20.2 per video on average) |
| Ratio of shots of interest | 25% (35,685 / 144,302) |
| Total length of shots of interest | 106.1 h |

* [[ted_raw_poses.zip]](https://drive.google.com/open?id=1vvweoCFAARODSa5J5Ew6dpGdHFHoEia2) 
[[z01]](https://drive.google.com/open?id=1zR-GIx3vbqCMkvJ1HdCMjthUpj03XKwB) 
[[z02]](https://drive.google.com/open?id=1B2SOnb_nTyJua9sII4w3xBjp5hBLRcAj) 
[[z03]](https://drive.google.com/open?id=1uhfv6k0Q3E7bUIxYDAVjxKIjPM_gL8Wm)
[[z04]](https://drive.google.com/open?id=1VLi0oQBW8xetN7XmkGZ-S_KhD-DvbVQB)
[[z05]](https://drive.google.com/open?id=1F2wiRX421f3hiUkEeKcTBbtsgOEBy7lh) (split zip files, total 80.9 GB)  
The result of Step 3. It contains the extracted human poses for all frames. 
* [[ted_shots_of_interest.zip, 13.3 GB]](https://drive.google.com/open?id=1kF7SVpxzhYEHCoSPpUt6aqSKvl9YaTEZ)  
The result of Step 6. It contains shot segmentation results ({video_id}.csv files) and shots of interest ({video_id}.json files). 
'clip_info' elements in JSON files have start/end frame numbers and a boolean value indicating shots of interest. 
The JSON files contain the extracted human poses for the shots of interest, 
so you don't need to download ted_raw_poses.zip unless the human poses for all frames are necessary.
* [[ted_gesture_dataset.zip, 1.1 GB]](https://drive.google.com/open?id=1lZfvufQ_CIy3d2GFU2dgqIVo1gdmG6Dh)  
The result of Step 8. Train/validation/test sets of speech-motion pairs. 
 
### Download videos and transcripts
We do not provide the videos and transcripts of TED talks due to copyright issues.
You should download actual videos and transcripts by yourself as follows:  
1. Download and copy [[video_ids.txt]](https://drive.google.com/open?id=1grFWC7GBIeF2zlaOEtCWw4YgqHe3AFU-) file which contains video ids into `./videos` directory.
2. Run `download_video.py`. It downloads the videos and transcripts in `video_ids.txt`.
Some videos may not match to the extracted poses that we provided if the videos are re-uploaded.
Please compare the numbers of frames, just in case.


## Citation 

If our code or dataset is helpful, please kindly cite the following paper:
```
@INPROCEEDINGS{
  yoonICRA19,
  title={Robots Learn Social Skills: End-to-End Learning of Co-Speech Gesture Generation for Humanoid Robots},
  author={Yoon, Youngwoo and Ko, Woo-Ri and Jang, Minsu and Lee, Jaeyeon and Kim, Jaehong and Lee, Geehyuk},
  booktitle={Proc. of The International Conference in Robotics and Automation (ICRA)},
  year={2019}
}
```

## Acknowledgement
* This work was supported by the ICT R&D program of MSIP/IITP. [2017-0-00162, Development of Human-care Robot Technology for Aging Society]   
* Thanks to [Eun-Sol Cho](https://github.com/euns2ol) and [Jongwon Kim](mailto:jwk9284@gmail.com) for contributions during their internships at ETRI.
