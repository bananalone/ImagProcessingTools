import argparse
from pathlib import Path

from common import print_args, copy_files


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, required=True, help='list of copyed files')
    parser.add_argument('-o', '--output', type=str, default=None, help='path to output')
    ### more arguments ###

    return parser.parse_args()


def list_files(file_path:str):
    with open(file_path, mode='r', encoding='utf-8') as f:
        content = f.read()
    files = content.strip().split()
    return files


def main(args):
    files = list_files(args.file)
    if not args.output:
        args.output = Path(args.file).parent
    copy_files(files, args.output)


if __name__ == '__main__':
    args = parse_args()
    print_args(args)
    main(args)