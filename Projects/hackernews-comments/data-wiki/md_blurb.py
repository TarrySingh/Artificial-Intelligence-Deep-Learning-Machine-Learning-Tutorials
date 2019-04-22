#!/usr/bin/env python3

import sys

def extract_blurb_sents(stream):
  prev_line = None
  blurb_sents = []
  for line in stream:
    line = line.strip()
    if prev_line is not None and line == '-' * len(prev_line):
      blurb_sents.pop()
      blurb_sents.pop()
      return blurb_sents
    blurb_sents.append(line)
    prev_line = line
  return blurb_sents

def sents_to_line(sents):
  return ' <P> '.join(sents)

if __name__ == '__main__':
  blurb_sents = extract_blurb_sents(sys.stdin)
  line = sents_to_line(blurb_sents)
  sys.stdout.write(line + '\n')
