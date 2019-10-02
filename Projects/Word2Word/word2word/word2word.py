#-*- coding: utf-8 -*-
from word2word.utils import download_or_load

class Word2word:
    def __init__(self, lang1, lang2):
        self.word2x, self.y2word, self.x2ys = download_or_load(lang1, lang2)

    def __call__(self, query, n_best=5):
        if query not in self.word2x:
            print("Sorry. There's no such word in the dictionary.")
        x = self.word2x[query]
        ys = self.x2ys[x]
        words = [self.y2word[y] for y in ys]
        return words[:n_best]
