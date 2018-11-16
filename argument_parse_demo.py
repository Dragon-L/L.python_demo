import argparse


parser = argparse.ArgumentParser()


def parse_position_argument():
    parser.add_argument('echo', type=int, choices=[1, 2, 3], help='echo the number of first parameter')
    args = parser.parse_args()
    print(args.echo)


def parse_optional_argument():
    parser.add_argument('-o', '--optional', help='this is a optional argument', default='argument')
    args = parser.parse_args()
    if args.optional:
        print('optional argument %s' % args.optional)


def parse_optional_argument_without_value():
    parser.add_argument('-o', '--optional', help='this is a optional argument', action='store_true')
    args = parser.parse_args()
    if args.optional:
        print('optional argument %s' % args.optional)


def define_conflict_argument():
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-v', action='store_true')
    group.add_argument('-f', action='store_true')
    args = parser.parse_args()
    if args.v:
        print('v')
    elif args.f:
        print('f')
    else:
        pass


# parse_position_argument()
# parse_optional_argument()
# parse_optional_argument_without_value()
define_conflict_argument()



