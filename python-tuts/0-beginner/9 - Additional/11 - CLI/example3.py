# let's fix that error by converting those strings to integers first

import sys

numbers = [int(a) for a in sys.argv[1:]]

print(sum(numbers))

# and call it as follows:
# python example3.py 1 2 3 4 5 6
# now we get the sum of the numbers passed in



