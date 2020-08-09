# while we can certainly use sys.argv this way, we often
# want more complicated sets of inputs:
# repeated values, different inputs, different types, optional inputs, etc

# we could maybe take this naive approach to get the first and last names
# from the input - to allow any order of specifying the input values
# we'll use the convention that the input name is prefixed by --
# and followed by the value itself

# something like: python example4.py --last-name Cleese --first-name John

import sys

for i in range(1, len(sys.argv), 2):
    print(sys.argv[i], sys.argv[i+1])

# we're getting somewhere, but now we have to make sense
# of the parameter names so we can assign the corresponding values
# to variables we can use in our program
