#!/bin/bash

grep '^INFO:tensorflow:loss = ' | sed -e 's/^.*loss = \([.0-9]*\), step = \([0-9]*\).*/\2\t\1/'
