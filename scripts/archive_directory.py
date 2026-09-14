#!/usr/bin/env python3
"""Create a ZIP archive atomically using the Python standard library."""

import argparse
import os
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    source = os.path.abspath(args.source)
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    temporary_base = f'{output}.tmp'
    temporary_zip = shutil.make_archive(
        temporary_base, 'zip', root_dir=os.path.dirname(source), base_dir=os.path.basename(source))
    os.replace(temporary_zip, output)


if __name__ == '__main__':
    main()
