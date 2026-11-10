#!/usr/bin/env python3
"""Repoint the research scripts at your local copy of the data.

The files in ``code/research/`` are the scripts exactly as they were run during the
study, so they still carry the authors' absolute data root
(``D:/DisasterM3/UrbanSARFloods``).  Run this once with your own path and they will
refer to your copy instead.

Usage
-----
    python 00_set_paths.py /path/to/UrbanSARFloods          # actually rewrite
    python 00_set_paths.py /path/to/UrbanSARFloods --dry-run # only report

The expected layout under the root is the one described in ``data/README.md``::

    UrbanSARFloods/
        urban_sar_floods/       # the downloaded dataset
        testing_case_256/       # evaluation patches
        <event>/testing_case_orig/<event>/<event>_GT.tif

Only plain-string replacements are done; no file is touched if it does not contain
the old prefix.
"""
import argparse
import os
import sys

OLD_POSIX = 'D:/DisasterM3/UrbanSARFloods'
OLD_WIN = r'D:\DisasterM3\UrbanSARFloods'


def main():
    ap = argparse.ArgumentParser(description='Repoint research scripts at a local data root.')
    ap.add_argument('root', help='path to your UrbanSARFloods directory')
    ap.add_argument('--dry-run', action='store_true', help='report only, do not write')
    args = ap.parse_args()

    new = os.path.abspath(args.root).replace('\\', '/')
    here = os.path.dirname(os.path.abspath(__file__))

    changed, scanned = 0, 0
    for name in sorted(os.listdir(here)):
        if not name.endswith('.py') or name == os.path.basename(__file__):
            continue
        path = os.path.join(here, name)
        with open(path, encoding='utf-8') as fh:
            text = fh.read()
        scanned += 1
        if OLD_POSIX not in text and OLD_WIN not in text:
            continue
        text = text.replace(OLD_WIN, new).replace(OLD_POSIX, new)
        changed += 1
        if args.dry_run:
            print('[dry-run] would rewrite', name)
        else:
            with open(path, 'w', encoding='utf-8', newline='\n') as fh:
                fh.write(text)
            print('rewrote', name)

    print(f'\nscanned {scanned} file(s), {"would rewrite" if args.dry_run else "rewrote"} {changed}')
    if changed and not args.dry_run:
        print('new data root:', new)
    return 0


if __name__ == '__main__':
    sys.exit(main())
