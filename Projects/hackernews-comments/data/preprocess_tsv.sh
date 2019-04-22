#!/bin/bash

mydir=$(dirname $BASH_SOURCE)

tokenize() {
  $mydir/mosesdecoder/scripts/tokenizer/normalize-punctuation.perl \
    | $mydir/mosesdecoder/scripts/tokenizer/tokenizer.perl \
        -protected $mydir/moses_tokenizer_protected.txt \
        -no-escape
}

lowercase() {
  $mydir/mosesdecoder/scripts/tokenizer/lowercase.perl
}

preprocess_comments() {
  $mydir/normalize_links.sh | tokenize
}

preprocess_titles() {
  $mydir/normalize_links.sh | tokenize | lowercase
}

cut -f4 $1.tsv | preprocess_comments > $1.pp.comments
cut -f3 $1.tsv | preprocess_titles > $1.pp.titles
