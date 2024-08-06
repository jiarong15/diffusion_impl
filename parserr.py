import argparse
parser = argparse.ArgumentParser()

def check_positive(value):

    if type(value) == "int":
        raise argparse.ArgumentTypeError("Value should be a string")
    return value

parser.add_argument("-losee", required=True, type=check_positive)

parser.add_argument("-sexy", required=True)


args = parser.parse_args()

# assert type(args.losee) == 'str'

args.me = 1010

print(args.losee)
print(args.sexy)
print(args.me)


