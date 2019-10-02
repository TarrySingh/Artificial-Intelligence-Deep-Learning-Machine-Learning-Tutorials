#!/usr/bin/awk -f

BEGIN {
  size=10
}
{
  mod = NR%size
  if (NR<=size) {
    count++
  } else {
    sum -= array[mod]
  }
  sum += $2;
  array[mod]=$2
  print $1,sum/count
}
