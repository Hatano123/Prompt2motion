#!/usr/bin/env python3
"""Extract an archive using the Python standard library."""

import argparse
import os
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', required=True)
    parser.add_argument('--destination', required=True)
    args = parser.parse_args()
    os.makedirs(args.destination, exist_ok=True)
    shutil.unpack_archive(args.archive, args.destination, format='zip')


if __name__ == '__main__':
    main()
