#!/usr/bin/env python3
"""
Do stuff
"""

import sys

from panflute import *

# Sections that we filter out completely
SECTION_IGNORE = [
  'See also',
  'Notes',
  'References',
  'External links',
  'Gallery',
  'Works cited',
  'External links and suggested reading',
  'Editions',
  'Discography',
  'Filmography',
  'Notes and references',
  'Further reading',
  'Works',
  'Awards',
  'Patents & awards',
]

# Elements that we ignore
ELEMENT_IGNORE = [
  Image,
  Note, # footnotes and endnotes
  Table,
  RawBlock,
  RawInline,
  SmallCaps,
  Strikeout,
  Subscript,
  Superscript,
]

def prepare(doc):
  doc.my_current_section = None

def action(elem, doc):
  #sys.stderr.write(repr(elem) + '\n')

  # We must not filter out Doc, that leads to errors
  if isinstance(elem, Doc):
    return elem

  # Filter certain sections
  if doc.my_current_section is not None and doc.my_current_section in SECTION_IGNORE:
    return []

  if isinstance(elem, Link):
    # For links, only keep the description text
    descr = elem.content.list

    # Filter links that don't have any description
    if len(descr) == 0:
      return []

    # Filter strange thumbnail links
    #
    # E.g. there is a link whose description is:
    # [Str(thumb|upright=1.1|),
    #  Emph(Str([The) Space Str(Astronomer](The_Astronomer_(Vermeer)) Space Str("wikilink"))),
    #  Space, Str(by), Space, Str([Johannes), Space, Str(Vermeer](Johannes_Vermeer), Space,
    #  Str("wikilink"))]
    link_str = stringify(elem)
    if (isinstance(descr[0], Str) and 'thumb|' in descr[0].text) or 'thumb|' in link_str:
      return []

    # Also ignore links to Wikipedia media
    if link_str.startswith('File:') or link_str.startswith('Category:'):
      return []

    return descr
  elif any([isinstance(elem, t) for t in ELEMENT_IGNORE]):
    return []
  elif isinstance(elem, Header):
    #if elem.level == 2:
    doc.my_current_section = stringify(elem)
    if doc.my_current_section in SECTION_IGNORE:
      return []
  elif isinstance(elem, Strong):
    # Hacker News only has Emph
    return Emph(*elem.content)
  elif isinstance(elem, Emph):
    # Hacker News only has Emph
    if len(elem.content) == 1 and isinstance(elem.content[0], Emph):
      return elem.content[0]
  else:
    return elem

def main(doc=None):
  return run_filter(action, prepare=prepare, doc=doc) 

if __name__ == '__main__':
  # If I don't do the following, I get:
	#
  # $ pandoc --wrap=none -f mediawiki -t markdown --filter filter_markdown.py < wiki/docs/00/New_Holland%2C_South_Dakota.txt
	# [...]
  #   File "[...]/panflute/elements.py", line 695, in __init__
  #     self.format = check_group(format, RAW_FORMATS)
  #   File "[...]/panflute/utils.py", line 34, in check_group
  #     raise TypeError(msg)
  # TypeError: element str not in group {'rtf', 'noteref', 'openxml', 'opendocument', 'latex', 'icml', 'html', 'context', 'tex'}
  # pandoc: Error running filter filter_markdown.py
  # Filter returned error status 1

  elements.RAW_FORMATS.add('mediawiki')
  main()
