#!/usr/bin/env python3

import sys

for fname in sys.stdin:
  with open(fname[:-1]) as f:
    for line in f:
      sys.stdout.write(line)
