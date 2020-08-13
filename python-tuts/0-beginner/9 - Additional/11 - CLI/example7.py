# let's try defining named arguments instead of just positional arguments
# (you can specify both types)

import argparse
import datetime

parser = argparse.ArgumentParser(description="Returns a string containing person's name and current age.")
parser.add_argument('-f', '--first', help='specify first name', type=str, required=False, dest='first_name')
parser.add_argument('-l', '--last', help='specify last name', type=str, required=True, dest='last_name')
parser.add_argument('--yob', help='year of birth', type=int, required=False, dest='birth_year')

args = parser.parse_args()

names = []
if args.first_name:
    names.append(args.first_name)

names.append(args.last_name)
full_name = ' '.join(names)

current_year = datetime.datetime.utcnow().year
age = current_year - args.birth_year

print(f'{full_name} is {age} years old.')

# Try running it as follows:
# python example7.py -h
# python example7.py -f Polly -l Parrot --yob 1969
# python example7.py -l Parrot --yob 196
# python example7.py -f Polly --yob 1969

