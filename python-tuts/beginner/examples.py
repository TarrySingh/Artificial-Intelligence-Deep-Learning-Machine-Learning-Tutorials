########################################################
# Full exercise kit on http://www.practicepython.org
#
#
#
########################################################
# / Some cool and funny examples //
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

########################################################
# Reverse word order solutions
########################################################

# 1 - Simple loop
def reverse(x):
    y = x.split()
    result = []
    for word in y:
        result.insert(0,word)
    return " ".join(result)
test1 = input('Enter your sentence:' )
print(reverse(test1))

# 2 - A quick one-liner solution is like this
def reverseSentence(x):
    return ''.join(x.split()[::-1])
enter = input('Your sentence goes here: ')
print(reverseSentence(enter))


########################################################
# Example 2 : Rock paper scissors game
########################################################

import sys

user1 = input('What is your name?')
user2 = input('and your name?')
user1_answer = input('%s, do you want to choose rock, paper or scissors?' %user1)
user2_answer = input('%s, do you want to choose rock, paper or scissors?' %user2)

def compare(u1, u2):
    if u1 == u2:
        return("Tts s tie!")
    elif u1 =='rock':
        if u2 == 'scissors':
            return('Rock wins!')
        else:
            return('Paper wins!')
    elif u1 =='scissors':
        if u2 == 'paper':
            return('Scissors wins!')
        else:
            return('Rock wins!')
    elif u1 =='paper':
        if u2 == 'rock':
            return('Paper wins!')
        else:
            return('Scissors wins!')
    else:
        return("Incorrect niput! You must enter rock, paper or scissors. Try one more time")
        sys.exit()

print(compare(user1_answer, user2_answer))
