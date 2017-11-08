import argparse
import os

import lmdb
from tqdm import tqdm


def get_keys(fn):
    with lmdb.open(fn, readonly=True, map_size=1099511627776) as env:
        with env.begin(buffers=True, write=False) as txn:
            it = txn.cursor().iternext(keys=True, values=False)
            for k in tqdm(it, total=txn.stat()['entries'], desc=fn):
                yield bytes(k)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+')
    args = parser.parse_args()

    for fn in args.files:
        if os.path.exists(fn + '.keys'):
            print("Skipping {}.keys".format(fn))
            continue
        with open(fn + '.keys.tmp', 'wb') as out:
            for key in get_keys(fn):
                out.write(key)
                out.write(b'\n')
        os.rename(fn + '.keys.tmp', fn + '.keys')


if __name__ == '__main__':
    main()
