#!/usr/bin/awk -f

# from: https://www.rosettacode.org/wiki/Count_occurrences_of_a_substring#AWK
function countsubstring(str, pat, len, i, c) {
  c = 0
  if (!(len = length(pat))) 
    return 0
  while (i = index(str, pat)) {
    str = substr(str, i + len)
    c++
  }
  return c
}

{
  if (NF > max_len)
    max_len = NF;

  counts[NF] += 1;
  paras[NF] += countsubstring($0, "<NL> <NL>") + 1
}
END {
  for (i = 1; i <= max_len; i++)
    printf "%d %.18f\n", i, paras[i]/counts[i]
}
