# sometimes we want to specify multiple values for a single argument

import argparse

parser = argparse.ArgumentParser('Prints the squares of a list of numbers, and the cubes of another list.')

parser.add_argument('--sq', help='list of numbers to square', nargs='*', type=float)
parser.add_argument('--cu', help='list of numbers to cube', nargs='+', type=float, required=True)

# here we are specifying that --sq may contain 0 or more elements (the *)
# but --cu must contain at least one element (the +)
# by default these two arguments are optional, but we could make them mandatory
# by setting required=True, which we do for the --cu argument (i.e. --cu
# specifies an argument that is mandatory (required=True) AND must have at least one value (nargs='+')

args = parser.parse_args()

# sq is optional, so we check to see if it is truthy (i.e. a non-empty list)
if args.sq:
    squares = [n ** 2 for n in args.sq]
    print(squares)

# cu we know is both mandatory and requires at least one value:
cubes = [n ** 3 for n in args.cu]
print(cubes)

# try the following:
# python example8.py -h
# python example8.py --sq 1.5 2 3 --cu 2.5 3 4
# python example8.py --sq --cu 2 3 4
# python example8.py --sq --cu
# python example8.py --sq 1 2 3



