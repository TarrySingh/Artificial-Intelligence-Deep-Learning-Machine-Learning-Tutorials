#!/usr/bin/env python2.7
# Split a Wikipedia XML dump into individual files. The files are stored in a
# directory tree based on hashing the title of the article.
#
# Evan Jones <evanj@mit.edu>
# April, 2008
# Released under a BSD licence.
# http://evanjones.ca/software/wikipedia2text.html
#
# Modified by:
# Leonard Dahlmann <leo.dahlmann@gmail.com>

import md5
import os
import sys
import urllib
import xml.sax

deletedTotal = 0
redirectsTotal = 0
bytesTotal = 0
bytesOut = 0
articleSkip = 0
articleWrite = 0
disambigTotal = 0
listsTotal = 0
log = open("xmldump2files.log", "a")

def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.2f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.2f%s%s" % (num, 'Yi', suffix)


def writeArticle(root, title, text):
    global articleSkip
    global articleWrite
    global deletedTotal
    global bytesOut
    global bytesTotal
    global log
    global redirectsTotal
    global disambigTotal
    global listsTotal
    # ~5.5 million articles at the moment
    # assuming an even distribution, we want 2 levels of 2 character directories:
    # 5.5 million / 256 / 256 = 83
    # Thus we won't have too many items in any directory

    title = title.encode("UTF-8")

    if "(disambiguation)" in title:
        disambigTotal = disambigTotal + 1
        return
    if title.startswith("List of"):
        listsTotal = listsTotal + 1
        return

    hash = md5.new(title).hexdigest()
    level1 = os.path.join(root, hash[0:2])

    # Wikipedia-ize the title for the file name
    title = title.replace(" ", "_")
    title = urllib.quote(title)
    # Special case for /: "%x" % ord("/") == 2f
    title = title.replace("/", "%2F")

    if len(title) > 123:
        title = hash

    title += ".txt"
    filename = os.path.join(level1, title)

    if text.startswith("#REDIRECT [[") or text.startswith("#REDIRECT[[") or text.startswith("#redirect [[") or text.startswith("#redirect[["):
        redirectsTotal = redirectsTotal + 1
        return

    encoded = text.encode("UTF-8")
    bytesTotal = bytesTotal + len(encoded)

    if not os.path.exists(filename):
        if not os.path.exists(level1):
            os.mkdir(level1)

        out = open(filename, "w")
        out.write(encoded)
        out.close()
        bytesOut = bytesOut + len(encoded)
        articleWrite = articleWrite + 1
    else:
        articleSkip = articleSkip + 1


    if (articleSkip + articleWrite) % 10000 == 0:
        percentComplete = (articleSkip + articleWrite) * 100 / 5500000
        string = "Redirects %d  Deleted %d  Disambigs %d  Lists %d  Skipped %d  Wrote %d %s  Total %d %s  (%d%%)\n" % (redirectsTotal, deletedTotal, disambigTotal, listsTotal, articleSkip, articleWrite, sizeof_fmt(bytesOut), articleWrite + articleSkip, sizeof_fmt(bytesTotal), percentComplete)
        log.write(string)
        log.flush()


class WikiPageSplitter(xml.sax.ContentHandler):
    def __init__(self, root):
        self.root = root
        self.stack = []
        self.text = None
        self.title = None

    def startElement(self, name, attributes):
        #~ print "start", name
        if name == "page":
            assert self.stack == []
            self.text = None
            self.title = None
        elif name == "title":
            assert self.stack == ["page"]
            assert self.title is None
            self.title = ""
        elif name == "text":
            assert self.stack == ["page"]
            assert self.text is None
            self.text = ""
        else:
            assert len(self.stack) == 0 or self.stack[-1] == "page"
            return

        self.stack.append(name)

    def endElement(self, name):
        #~ print "end", name
        if len(self.stack) > 0 and name == self.stack[-1]:
            del self.stack[-1]
        if name == "text":
            # We have the complete article: write it out
            writeArticle(self.root, self.title, self.text)

    def characters(self, content):
        assert content is not None and len(content) > 0
        if len(self.stack) == 0:
            return

        if self.stack[-1] == "title":
            self.title += content
        elif self.stack[-1] == "text":
            assert self.title is not None
            self.text += content


xml.sax.parse(sys.argv[1], WikiPageSplitter(sys.argv[2]))

log.write("done\n")
log.close()
