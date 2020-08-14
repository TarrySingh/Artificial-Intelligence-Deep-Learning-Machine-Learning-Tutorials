# in this example we see how to set up default values if the argument is not specified
# and we also see how to implement flags (i.e. the argument name is required, but it does not have a value)
# we also introduce the action parameter for arguments

# when we define arguments, we can associate an action to be taken when the argument is parsed
# by default, the action is store, which simply stores the value (if any) in an attribute
# of the same name, or as defined by dest as we saw earlier

# but other actions are possible too:
#   - store_const: this is used in conjunction with the const parameter which defines the value
#       to store for that argument (will disallow any values passed on the command line for that arg),
#       so useful for flags
#   - often flags are simply True/False, in which case we can use 'store_true' and 'store_false'
# and more... In fact, we can even define our own custom actions
# see https://docs.python.org/3/library/argparse.html#action
# and for more info if you're interested

import argparse

parser = argparse.ArgumentParser(description='testing defaults and flags')

parser.add_argument('--monty', action='store_const', const='Python')
parser.add_argument('-v', '--verbose', action='store_const', const=True, default=True)
parser.add_argument('-v2', '--verbose2', action='store_const', const=True)  # no default!
parser.add_argument('-q', '--quiet', action='store_false')

parser.add_argument('-n', '--name', default='John', type=str)

args = parser.parse_args()

print(args)

# Try the following:
# python example9.py -h
# python example9.py
# python example9.py --monty -v -v2 -q -n Eric