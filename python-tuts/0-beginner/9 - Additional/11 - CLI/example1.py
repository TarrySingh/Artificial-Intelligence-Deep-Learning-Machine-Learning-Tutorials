# when running a python script from the command line
# we can pass arguments to that program
# these arguments can be found in the sys.argv property

# This property is a list containing a sequence of strings
# where the first string is the name of the python file
# that was invoked, and the subsequent strings are the parameter names
# and values passed on the command line



import sys

print(sys.argv)

# try running this script as follows:
# python example1.py 123 hello 456 goodbye
# Output:
# ['example1.py', '123', 'hello', '456', 'goodbye']

# now try running it this way:
# python example1.py [1, 2, 3] [4, 5, 6]
# Output:
# ['example1.py', '[1,', '2,', '3]', '[4,', '5,', '6]']

# or even this example:
# python example1.py --name John --years 1980 1981 1982
# Output:
# ['example1.py', '--name', 'John', '--years', '1980', '1981', '1982']



