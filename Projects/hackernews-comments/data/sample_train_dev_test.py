#!/usr/bin/env python3
"""
Split a title-comments file into train/dev/test randomly,
while making sure that the same title only occurs in one of the sets.

Reads TSV from STDIN, assuming that it is ordered (or grouped) by the title.
"""

import argparse
import sys
import random

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--train',
                    required=True,
                    help='output file for the training data')
parser.add_argument('--dev',
                    required=True,
                    nargs=2,
                    metavar=('DEV-FILE', 'DEV-PERCENTAGE'),
                    help='output file for the dev data and the percentage of data assigned for dev (space separated)')
parser.add_argument('--test',
                    required=True,
                    nargs=2,
                    metavar=('TEST-FILE', 'TEST-PERCENTAGE'),
                    help='output file for the test data and the percentage of data assigned for test (space separated)')
parser.add_argument('--seed',
                    type=int,
                    help='initialization for the random number generator (default: no fixed seed)')
args = parser.parse_args()

if args.seed:
  random.seed(args.seed)

dev_perc = float(args.dev[1])
test_perc = float(args.test[1])
train_perc = 100.0 - dev_perc - test_perc

sys.stderr.write('sampling train at {:.4f}%, dev at {:.4f}%, test at {:.4f}%\n'
  .format(train_perc, dev_perc, test_perc))

with open(args.train, 'w') as f_train, \
     open(args.dev[0], 'w') as f_dev, \
     open(args.test[0], 'w') as f_test:
  choices = [f_train, f_dev, f_test]
  weights = [train_perc, dev_perc, test_perc]

  # The input is assumed to be sorted (or grouped) by title.
  # Accumulate comments for the current title until we move on to the next
  # title or reach the end, then decide which to file to write into.
  current_title = None
  current_lines = []

  def flush():
    [choice] = random.choices(choices, weights)

    for line in current_lines:
      choice.write(line)

  for line in sys.stdin:
    cols = line.split('\t')
    new_title = cols[2]

    if current_title is None:
      current_title = new_title
    elif new_title != current_title:
      flush()
      current_title = None
      current_lines.clear()

    current_lines.append(line)

  flush()
