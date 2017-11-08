# Based vaguely on
# https://github.com/tensorflow/models/blob/5f0776a23de46668634ee0e8ca4e73446ff608c1/research/real_nvp/lsun_formatting.py
# and on
# https://github.com/tensorflow/models/blob/5f0776a23de46668634ee0e8ca4e73446ff608c1/research/slim/datasets/build_imagenet_data.py
# which have the following license:

# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from io import BytesIO
import itertools
import math
import os

import lmdb
import numpy as np
from PIL import Image
from tqdm import tqdm
from six.moves import range, zip
import tensorflow as tf


LABELS = {}
LABEL_NAMES = [None] * 10
with open('category_indices.txt') as f:
    for line in f:
        n, i = line.split()
        LABELS[n] = int(i)
        LABEL_NAMES[int(i)] = n


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def writers_per_file(name_fmt, n_per_file, bar=None, verbose=True, start=0):
    for f_num in itertools.count(start):
        fname = name_fmt.format(f_num)
        if bar is None and verbose:
            print("Writing to {}".format(fname))
        else:
            bar.set_postfix({'file': os.path.basename(fname)})
        with tf.python_io.TFRecordWriter(fname) as writer:
            for i in range(n_per_file):
                yield writer


def image_stream(files, order, skips=None):
    envs = [lmdb.open(f, readonly=True, map_size=1099511627776) for f in files]
    if skips is None:
        skips = [0] * len(files)
        skip_tos = [None] * len(files)
    else:
        skips = [int(s) for s in skips]
        skip_tos = [None] * len(files)
        for i, (f, skip) in enumerate(zip(files, skips)):
            if skip > 0 and os.path.isfile(f + '.keys'):
                with open(f + '.keys', 'rb') as keys:
                    assert len(keys.readline()) == 41
                    keys.seek(skip * 41)
                    l = keys.readline()
                    assert len(l) == 41
                    assert l[-1:] == b'\n'
                    skip_tos[i] = l[:-1]
                    # skip_tos[i] = next(itertools.islice(keys, skip, None))
                    skips[i] = 0

    streams = [image_files(env, skip=s, skip_to=k)
               for env, s, k in zip(envs, skips, skip_tos)]
    for i in order:
        yield (files[i],) + next(streams[i])


def image_files(env, skip=0, skip_to=None):
    with env.begin(write=False, buffers=True) as txn:
        cur = txn.cursor()
        if skip_to is not None:
            cur.set_key(skip_to)
        elif skip:
            for _ in range(skip):
                cur.next()
                # XXX this is slow. LMDB doesn't seem to have a faster option.
        for key, val in iter(cur):
            with BytesIO(val) as f:
                yield bytes(key), f


_channels = _int64_feature(3)
_rgb = _bytes_feature('RGB'.encode('utf-8'))
_jpeg = _bytes_feature('JPEG'.encode('utf-8'))


def _ceil(x):
    return int(math.ceil(x))


def squarify(width, height):
    wcent = width // 2
    hcent = height // 2
    if height < width:
        return (wcent - height // 2, 0, wcent + _ceil(height / 2), height)
    else:
        return (0, hcent - width // 2, width, hcent + _ceil(width / 2))


def convert_file(im_bio, name, db_name, label,
                 resize=None, crop_square=False):
    im = Image.open(im_bio)
    changed = False
    if im.format != 'JPEG' or im.mode != 'RGB':
        im = im.convert('RGB')
        changed = True

    if resize is not None:
        if min(im.height, im.width) != resize:
            if crop_square:
                im = im.resize((resize, resize), resample=Image.LANCZOS,
                               box=squarify(im.width, im.height))
            else:
                sz = np.array([im.width, im.height], dtype=float)
                new = (resize * sz / sz.min()).round().astype(int)
                im = im.resize(tuple(new))
            changed = True
        elif crop_square:
            im = im.crop(box=squarify(im.width, im.height))
            changed = True
    elif crop_square and im.height != im.width:
        im = im.crop(box=squarify(im.width, im.height))
        changed = True

    if changed:
        im_bio = BytesIO()
        im.save(im_bio, format='JPEG')

    return tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(im.height),
        'image/width': _int64_feature(im.width),
        'image/colorspace': _rgb,
        'image/channels': _channels,
        'image/class/label': _int64_feature(label),
        'image/filename': _bytes_feature(name),
        'image/db': _bytes_feature(db_name.encode('utf-8')),
        'image/format': _jpeg,
        'image/encoded': _bytes_feature(im_bio.getvalue()),
    }))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('out_base')
    parser.add_argument('--cats', nargs='+', default=LABEL_NAMES)
    parser.add_argument('--n-per-file', type=int, default=100000)
    parser.add_argument('--order-seed', type=int, default=1)
    parser.add_argument('--set', choices=['train', 'val', 'test'],
                        default='train')
    parser.add_argument('--resize', type=int)
    parser.add_argument('--crop-square', action='store_true', default=False)
    parser.add_argument('--n-jobs', '-J', type=int, default=1)
    parser.add_argument('--job-num', '-j', type=int, default=0)
    args = parser.parse_args()
    if args.job_num >= args.n_jobs:
        parser.error("JOB_NUM must be less than N_JOBS")
    if args.job_num < 0:
        parser.error("JOB_NUM must be at least 0")

    if args.set == 'test':
        files = ['test.lmdb']
    else:
        files = ['{}_{}.lmdb'.format(n, args.set) for n in args.cats]

    if not os.path.isdir(os.path.dirname(args.out_base)):
        os.makedirs(os.path.dirname(args.out_base))

    ns = [lmdb.open(f, readonly=True).stat()['entries'] for f in files]
    n_total = sum(ns)

    order = np.empty(n_total, dtype=np.uint8)
    start = 0
    for i, n in enumerate(ns):
        order[start:start + n] = i
        start += n
    rs = np.random.RandomState(args.order_seed)
    rs.shuffle(order)

    total_files = _ceil(n_total / args.n_per_file)
    files_per_job = _ceil(total_files / args.n_jobs)
    first_file = args.job_num * files_per_job
    first_img = first_file * args.n_per_file
    last_img = min(n_total, first_img + args.n_per_file * files_per_job) - 1

    if args.n_jobs != 1:
        print("Doing job {} of {}:".format(args.job_num, args.n_jobs), end='\t')
        print("images {} - {}".format(first_img, last_img), end='\t')
        print("files {} - {}".format(first_file, last_img // args.n_per_file))

    skips = np.bincount(order[:first_img], minlength=len(files))
    images = image_stream(files, order[first_img:last_img], skips=skips)

    bar = tqdm(range(last_img - first_img + 1))

    writer_fmt = (args.out_base + '_{:0' + str(len(str(total_files - 1)))
                  + '}.tfrecords')
    writers = writers_per_file(writer_fmt, args.n_per_file, bar=bar,
                               start=first_file)

    label_lookup = {}
    for f in files:
        if args.set == 'test':
            label_lookup[f] = -1
        assert f.endswith('.lmdb')
        cat, s = f[:-5].split('_')
        assert s in {'train', 'val'}
        label_lookup[f] = LABELS[cat]

    for i, writer, (db_name, name, im_file) in zip(bar, writers, images):
        ex = convert_file(im_file, name=name,
                          db_name=db_name, label=label_lookup[db_name],
                          resize=args.resize, crop_square=args.crop_square)
        writer.write(ex.SerializeToString())


if __name__ == "__main__":
    main()
