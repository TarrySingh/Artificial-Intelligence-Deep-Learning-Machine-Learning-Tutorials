# so we can use these values that are passed in that sys.argv list
# to receive input values for our application

# let's write a simple sum application

import sys

numbers = sys.argv[1:]

print(sum(numbers))

# and call it as follows:
# python example2.py 1 2 3 4 5 6
# we get an error! that's because these arguments are always considered strings

