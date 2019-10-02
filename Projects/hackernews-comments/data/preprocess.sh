#!/bin/bash

mydir=$(dirname $BASH_SOURCE)

$mydir/normalize_links.sh \
  | $mydir/mosesdecoder/scripts/tokenizer/normalize-punctuation.perl \
  | $mydir/mosesdecoder/scripts/tokenizer/tokenizer.perl \
      -protected $mydir/moses_tokenizer_protected.txt \
      -no-escape
