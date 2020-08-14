# sometimes we want to make two (or more) arguments mutually exclusive,
# i.e. we cannot specify both at once
# for example, we may have something where we want the user to specify verbose output,
# quiet output, or neither, but not both

import argparse
import cmath


parser = argparse.ArgumentParser()

group = parser.add_mutually_exclusive_group()
group.add_argument('-v', '--verbose', action='store_true')
group.add_argument('-q', '--quiet', action='store_true')


parser.add_argument('-n', type=complex, help='some complex number', required=True)

args = parser.parse_args()

if args.quiet:
    print('quiet mode...')
    print('nothing to see here.')
elif args.verbose:
    print('verbose mode...')
    print(f'input: {args.n}')
    print(f're={args.n.real}, im={args.n.imag}')
    print(f'{args.n} = {cmath.polar(args.n)}')
else:
    print('normal mode...')
    print(f'{args.n} = {cmath.polar(args.n)}')


# try calling it:
# python example10.py -h
# python example10.py -q -n 3+4j
# python example10.py -v -n 3+4j
# python example10.py -n 3+4j
