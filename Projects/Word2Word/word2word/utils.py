#-*- coding: utf-8 -*-
import platform
import wget
import requests
import logging
import os
import pickle


def get_savedir():
    pf = platform.system()
    if pf == "Windows":
        savedir = "C:\word2word"
    elif pf == "Linux":
        savedir = "/usr/share/word2word"
    else:
        savedir = "/usr/local/share/word2word"

    if not os.path.exists(savedir):
        os.makedirs(savedir, exist_ok=True)
    return savedir


def exists(path):
    r = requests.head(path)
    return r.status_code == requests.codes.ok


def get_download_url(lang1, lang2):
    filepath = os.path.dirname(os.path.abspath(__file__)) + '/supporting_languages.txt'
    for line in open(filepath, 'r'):
        l1, l2 = line.strip().split("-")
        if lang1 == l1 and lang2 == l2:
            return f"https://mk.kakaocdn.net/dn/kakaobrain/word2word/{lang1}-{lang2}.pkl"
    raise Exception("Not supperted language")


def download_or_load(lang1, lang2):
    savedir = get_savedir()
    fpath = os.path.join(savedir, f"{lang1}-{lang2}.pkl")
    if not os.path.exists(fpath):
        # download from cloud
        url = get_download_url(lang1, lang2)
        if url is None:
            raise ValueError("There's no data for those languages")

        if not exists(url):
            raise ValueError("Sorry. There seems to be some problem in the cloud access.")

        logging.info("Download data ...")
        wget.download(url, fpath)
    word2x, y2word, x2ys = pickle.load(open(fpath, 'rb'))
    return word2x, y2word, x2ys
