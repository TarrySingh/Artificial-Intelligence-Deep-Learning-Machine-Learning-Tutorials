#!/bin/bash

mydir=$(dirname $BASH_SOURCE)

lowercase() {
  $mydir/../data/mosesdecoder/scripts/tokenizer/lowercase.perl
}

preprocess_titles() {
  lowercase
}

cut -f1 $1.tsv | preprocess_titles > $1.pp.titles
cut -f2 $1.tsv > $1.pp.comments
