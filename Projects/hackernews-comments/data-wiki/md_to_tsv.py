#!/usr/bin/env python3

import sys
import os
import urllib.parse

class Node(object):
  def __init__(self, header, level=1, parent=None):
    self.header = header
    self.level = level
    self.parent = parent
    self.children = []
    self.text = []

  def add_child(self, header):
    child = Node(header=header, level=self.level+1, parent=self)
    self.children.append(child)
    return child

  def write(self, out):
    for line in self.text:
      out.write(line)
      out.write('\n')

    for child in self.children:
      out.write('\n')
      out.write('#' * (self.level+1))
      out.write(' ')
      out.write(child.header)
      out.write('\n')
      child.write(out)

  def write_tsv(self, out, header_path=None):
    if header_path is None:
      my_header_path = self.header
    else:
      my_header_path = header_path + ': ' + self.header
    
    if self.text:
      out.write(my_header_path)
      out.write('\t')
      for i in range(len(self.text)-1):
        out.write(self.text[i].strip())
        out.write(' <NL> <NL> ')
      out.write(self.text[-1])
      out.write('\n')

    for child in self.children:
      child.write_tsv(out, my_header_path)

def header_level(string):
  i = 0
  while i < len(string) and string[i] == '#':
    i += 1

  if i+1 >= len(string) or string[i] != ' ':
    return (0, None)

  return (i, string[i+1:])

def parse_md_tree(stream, title):
  root = Node(header=title)
  cur = root

  for line in stream:
    if line[-1] == '\n':
      line = line[:-1]

    level_header = None
    if cur.text \
        and len(line) == len(cur.text[-1]) \
        and all(c == '-' for c in line):
          level_header = (2, cur.text.pop())
    else:
      level_header = header_level(line)

    if level_header[0] > 0 and level_header[1] is not None:
      #assert not cur.text or not cur.text[-1]
      #if cur.text:
        #cur.text.pop()

      if level_header[0] > cur.level + 1:
        raise ValueError('Invalid header level: {} (header: {}) vs current {} (header: {})'
            .format(level_header[0], level_header[1], cur.level, cur.header))
      
      while cur.level+1 != level_header[0]:
        cur = cur.parent

      cur = cur.add_child(header=level_header[1].strip())
    elif len(line) > 0: #or cur.text:
      cur.text.append(line.replace('\n', ' ').replace('\t', ' '))

  return root

if __name__ == '__main__':
  filename = sys.argv[1]
  basename = os.path.splitext(os.path.basename(filename))[0]

  title = urllib.parse.unquote(basename.replace('_', ' '))

  tree = parse_md_tree(sys.stdin, title)
  #tree.write(sys.stdout)
  tree.write_tsv(sys.stdout)
