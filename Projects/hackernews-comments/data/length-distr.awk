#!/usr/bin/awk -f

{
  if (NF > max_len)
    max_len = NF

  counts[NF] += 1
  total += 1
}
END {
  for (i = 1; i <= max_len; i++)
    printf "%d %.18f\n", i, counts[i]/total
}
