#!/usr/bin/env python
""" @namespace doxypy
doxypy is an input filter for Doxygen. It preprocesses python 
files so that docstrings of classes and functions are extracted 
as Doxygens special python documentation blocks. It can be found
at <http://code.foosel.net/doxypy>.

Copyright (C) 2006  
	Gina Haeussge (gina at foosel dot net),
	Philippe Neumann (phil at foosel dot net)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""

""" @mainpage
@author Gina Haeussge
@author Philippe Neumann
"""

import sys
import re
from optparse import OptionParser, OptionGroup

def makeCommentBlock(commentLines, indent=""):
	"""	Converts the given $commentLines into a comment block.
		
		@param	commentLines	The lines of the block comment
		@param	indent			The indentation of the block
		
		@return The indented doxygen comment block containing the 
				given comment lines.
	"""
	
	joinStr = "\n%s# " % indent
	
	if options.strip:
		commentLines = map(lambda x: x.strip(), commentLines)

	lines = joinStr.join(commentLines)
	
	return "%s##%s%s" % (indent, joinStr, lines)

def parse(input):
	""" Searches for def and class definitions in the source, then moves
		existing docstrings as special doxygen block comments in front of the
		definitions.
		
		@param	input	The input source to process
		
		@return The processed input.
	"""
	
	output = []
	
	# Comment delimiter of the docstring
	commentDelim = '"""'
	
	# Some regexes
	triggerRe = re.compile("^(\s*)(def .+:|class .+:)")
	commentStartRe = re.compile('^\s*(%s)' % commentDelim)
	commentEndRe = re.compile('(%s)\s*$' % commentDelim)
	emptyRe = re.compile("^\s*$")
	hashLineRe = re.compile("^\s*#.*$")
	importLineRe = re.compile("^\s*(import |from .+ import)")
	
	# split input into lines
	lines = input.split("\n")
	
	# flags, buffers, ...
	fileHeadFlag = True
	triggerWordFlag = False
	commentFlag = False
	comment = []
	triggerWs = ""
	triggerLines = None
	
	# process each line
	for line in enumerate(lines):

		match = re.search(triggerRe, line[1])
		if match:
			if triggerWordFlag and triggerLines:
				output.append("\n".join(triggerLines))
			
			triggerWordFlag = True
			triggerWs = match.group(1)
			fileHeadFlag = False
			triggerLines = [line[1]]
			continue

		# file header or active keyword trigger?
		if fileHeadFlag or triggerWordFlag:
			# comment end of multiline comment found
			if re.search(commentEndRe, line[1]) and commentFlag:
				comment.append( line[1][ : line[1].rfind(commentDelim) ] )
				output.append(makeCommentBlock(comment, triggerWs))
				if triggerLines:
					output.append("\n".join(triggerLines))
				comment = []
				commentFlag = False
				triggerWs = ""
				triggerLines = None
				triggerWordFlag = False
				
			# comment start found
			elif re.search(commentStartRe, line[1]):
	
				if re.search(commentEndRe, line[1][line[1].find(commentDelim)+len(commentDelim) :]):
					# singleline comment
					comment.append(line[1][line[1].find(commentDelim)+len(commentDelim) : line[1].rfind(commentDelim)])
					output.append(makeCommentBlock(comment, triggerWs))
					
					if triggerLines:
						output.append("\n".join(triggerLines))
						
					comment = []
					commentFlag = False
					triggerWs = ""
					triggerLines = None
					triggerWordFlag = False
					
				else:
					# multiline comment begin
					commentFlag = True
					comment.append(
						line[1][line[1].find(commentDelim)+len(commentDelim):]
					)
	
			# active multiline comment -> append comment
			elif commentFlag:
				comment.append(line[1])
			
			# still searching for comment
			elif re.search(emptyRe, line[1]):
				if triggerLines:
					triggerLines.append(line[1])
				else:
					output.append(line[1])
			
			# searching for file header
			elif fileHeadFlag:
				if not (re.search(hashLineRe, line[1]) or re.search(emptyRe, line[1]) or re.search(importLineRe, line[1])):
					# fileheader over -> disable search
					fileHeadFlag = False
				output.append(line[1])
			
			# no comment, disable comment search mode
			else:
				triggerWordFlag = False
				if triggerLines:
					output.append("\n".join(triggerLines))
				triggerLines = None
				output.append(line[1])
		
		# just append the line
		else:
			output.append(line[1])
	
	# return output
	return "\n".join(output)
	
def loadFile(filename):
	"""	Loads file $filename and returns the content.
	
		@param	filename	The name of the file to load
		@return The content of the file.
	"""
	
	f = open(filename, 'r')
	
	try:
		content = f.read()
		return content
	finally:
		f.close()
		
def optParse():
	"""parses commandline options"""
	
	parser = OptionParser(prog="doxypy", version="%prog 0.2.1")
	
	parser.set_usage("%prog [options] filename")
	parser.add_option("--trim", "--strip",
		action="store_true", dest="strip",
		help="enables trimming of docstrings, might be useful if you get oddly spaced output"
	)
	
	## parse options
	global options
	(options, filename) = parser.parse_args()
		
	if not filename:
		print >>sys.stderr, "No filename given."
		sys.exit(-1)
	
	return filename[0]

def main():
	""" Opens the file given as first commandline argument and processes it,
		then prints out the processed file contents.
	"""
	
	filename = optParse()
	
	try:
		input = loadFile(filename)
	except IOError, (errno, msg):
		print >>sys.stderr, msg
		sys.exit(-1)
	
	output = parse(input)
	print output
	
if __name__ == "__main__":
	main()