########################################################
# Full exercise kit on http://www.practicepython.org
#
#
#
########################################################
# / Some password generator example //
########################################################

########################################################
# Example 1 - Password Generator
########################################################
# 1 - Basic example

import random
import string

# s = string.printable
# passlen = 8
#
# p = "".join(random.sample(s,passlen))
# print(p)

# 2 - Somewhat more fun

def pw_gen(size = 8, chars=string.printable):
    return ''.join(random.choice(chars) for _ in range(size))
# print('Password is '+ pw_gen())
print('Password is '+ pw_gen(int(input('How many characters in your password?'))))
