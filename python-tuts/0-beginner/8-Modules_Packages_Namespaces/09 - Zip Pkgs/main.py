import sys
sys.path.append('./common.zip')

import common
import common.validators as validators
import common.models as models
# from common.models import *
import common.helpers as helpers



validators.is_boolean('true')
validators.is_json('{}')
validators.is_numeric(10)
validators.is_date('2018-0101')

john_post = models.Post()
john_posts = models.Posts()
john = models.User()


print('\n\n***** self *****')
for k in dict(globals()).keys():
    print(k)


print('\n\n***** common *****')
for k in common.__dict__.keys():
    print(k)

print('\n\n***** models *****')
for k in common.models.__dict__.keys():
    print(k)

print(helpers.say_hello('Python'))
print(helpers.factorial(5))

