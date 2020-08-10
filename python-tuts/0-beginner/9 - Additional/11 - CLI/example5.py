# Maybe something like this:

import sys

# let's create a dictionary of the name/value pairs

# the parameter names
keys = sys.argv[1::2]
values = sys.argv[2::2]
print(keys)
print(values)

# next create a dictionary so we can easily look up the value for a given key
args = {k: v for k, v in zip(keys, values)}
print(args)

# finally let's assign the arguments to variables:
first_name = args.get('--first-name')
last_name = args.get('--last-name')
print(first_name, last_name)

# call it this way:
# python example5.py --last-name Cleese --first-name John

# so we can certainly take this approach, but it's going to
# get quite tedious, and we really want more functionality
# maybe something like:
# --name parrot --states dead late no-more deceased --year 1969
# and we would want name to be a string, states to be a list, and year to be an int
# additionally we want to make some of them optional, some mandatory, allow
# users to specify both long and short argument names, have some
# arguments that have names but no values, oh, and provide help
# when the user runs the script using a -h or --help argument!

# Fortunately all that heavy lifting has been done for us in the...
# yep, standard library. In particular, the argparse module.

