#!/usr/bin/gawk -f

BEGIN {
  FS="\t"
} {
  if ((NR-1) % 5 == 0)
    print "## " $1
  gsub(/<NL>/, "\n  ", $2)
  gsub(/\\ -/, " -", $2)
  o = gensub(/_ ([^_]*) _/, "_\\1_", "g", $2)
  print "- " o
  print "\n"
}
