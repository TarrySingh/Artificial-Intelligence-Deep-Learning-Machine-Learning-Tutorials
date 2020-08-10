# simple use of argparse module

# to parse command line parameters, we need to set up a parser object
# and we tell it a few things:
# 1. a description of what the program does that will be displayed
#       when the argument is -h or --help. The arg parser will
#       automatically add help information for the different
#       arguments that are supported
# 2. the arguments we want to support, which can be a mix of positional
#       or named arguments, some help text about each one, and even the expected data type

import argparse

# this sets up the parser, defining two positional parameters
parser = argparse.ArgumentParser('Calculates the div a//b and mod a % b of two integers.')
parser.add_argument("a", help='first integer', type=int)
parser.add_argument("b", help='second integer', type=int)

# this actually reads sys.argv and parses the data contained therein

# so we could do it this way:
# import sys
# print(sys.argv)
# args = parser.parse_args(sys.argv[1:])

# or simply:
args = parser.parse_args()

a = args.a
b = args.b

print(f'{a}//{b} = {a//b}, {a}%{b} = {a%b}')

# now try running these:
# python example6.py -h
# python example6.py 10 3
# python example6.py 10.5 3

# pretty neat!