#!/bin/bash

awk -F '\\|\\|\\|' '/\|\|\|/ { s+=$1; m=split($2,a," "); n+=m+1; } END { print exp(s / n) }'
