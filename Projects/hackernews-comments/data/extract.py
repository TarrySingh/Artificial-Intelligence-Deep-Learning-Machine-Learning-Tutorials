#!/usr/bin/env python3
"""
Extract only the top-level comments from a HN data dump.

The dump is read from STDIN in the JSON format (see README.md).
Extracted comments are written to STDOUT in TSV, with the following columns:
1. ID
2. Time
3. Story Title
4. Text

Statistics are written on STDERR.
"""

import argparse
import sys
import json
import html2text

h2t = html2text.HTML2Text()
h2t.body_width = 0 # don't automatically wrap when converting to Markdown

def normalize_text(text):
  # The data dump seems to contain carriage return frequently
  text = text.replace('\r', '')

  # In some examples, it seems that double newline is used for paragraphs,
  # other examples use <p>. For html2text, we need the latter.
  text = text.replace('\n\n', '<p>')

  # We should be fine with ignoring the remaining single newlines
  text = text.replace('\n', ' ')

  # HTML -> Markdown
  text = h2t.handle(text)

  # There are some trailing newlines in the markdown
  text = text.strip()
  text = text.replace('\r', '')

  # Finally, convert whitespace so that we can give line-by-line tab separated output
  text = text.replace('\t', ' ')
  text = text.replace('<NL>', ' ') # these are texts written by programmers,
                                   # but let's not bother with this special case
  text = text.replace('\n', ' <NL> ')

  return text

class Converter(object):
  def __init__(self):
    # We remember a mapping from object IDs to the titles of stories.
    # This way we can tell if a comment is top-level by checking if its parent is a story.
    # This works in a single pass since the input lines are sorted by ID.
    self.story_titles = {}

    # Let's keep some stats
    self.n_total = 0
    self.n_comments = 0
    self.n_top_level_comments = 0
    self.n_unexpected_format = 0
    self.n_ignored = 0
    self.n_deleted = 0
 
  def _process_object(self, body, f_out):
    object_type = body['type']

    if object_type == 'story':
      title = body['title'].strip()
      title = title.replace('\r', '')
      title = title.replace('\n', ' ')
      title = title.replace('\t', ' ')

      if len(title) == 0:
        self.n_ignored += 1
        return

      self.story_titles[body['id']] = title
    elif object_type == 'comment':
      story_title = self.story_titles.get(body['parent'])

      if story_title is not None:
        # Yay, got a top-level comment!

        text = normalize_text(body['text'])
        if len(text) == 0:
          self.n_ignored += 1
          return

        f_out.write(str(body['id']))
        f_out.write('\t')
        f_out.write(str(body['time']))
        f_out.write('\t')
        f_out.write(story_title)
        f_out.write('\t')
        f_out.write(normalize_text(body['text']))
        f_out.write('\n')

        self.n_top_level_comments += 1

      self.n_comments += 1 

    else:
      # Probably object_type == 'job'
      self.n_ignored += 1
      pass

  def process_object(self, obj, f_out):
    try:
      self.n_total += 1

      body = obj['body']

      if body.get('deleted') == True:
        self.n_deleted += 1
        return

      # Some of the titles contain "algolia" as well as "site" fields,
      # in which the actual "body" is stored
      algolia = body.get('algolia')
      if algolia is not None:
        body = algolia

      site = body.get('site')
      if site is not None:
        body = site

      # Those titles that have their body in "site" don't always have the
      # "id", let's copy it over if possible
      if 'id' not in body and 'id' in obj:
        body['id'] = obj['id']

      self._process_object(body, f_out)
    except KeyError as e:
      # Not sure why this happens, but a few lines in the input seem to be missing fields
      self.n_unexpected_format += 1

  def write_stats(self, f_out):
    f_out.write('stories:\t{}\n'.format(len(self.story_titles)))
    if len(self.story_titles) > 0:
      f_out.write('comments:\t{} ({:.2f} per title)\n'.format(self.n_comments, self.n_comments / float(len(self.story_titles))))
    if self.n_comments > 0:
      f_out.write('top-level:\t{} ({:.4f}%)\n'.format(self.n_top_level_comments, self.n_top_level_comments / float(self.n_comments) * 100.0))

    f_out.write('ignored rows:\t{:.4f}%\n'.format(self.n_ignored / float(self.n_total) * 100.0))
    f_out.write('invalid rows:\t{:.4f}%\n'.format(self.n_unexpected_format / float(self.n_total) * 100.0))
    f_out.write('deleted rows:\t{:.4f}%\n'.format(self.n_deleted / float(self.n_total) * 100.0))

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  args = parser.parse_args()

  converter = Converter()

  for n, line in enumerate(sys.stdin):
    converter.process_object(json.loads(line), f_out=sys.stdout)
    if (n+1) % 500_000 == 0:
      sys.stderr.write('[{:.1f}M]\n'.format((n+1) / float(1_000_000)))
      converter.write_stats(sys.stderr)

  converter.write_stats(f_out=sys.stderr)
