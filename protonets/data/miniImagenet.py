import os
import sys
import glob

from functools import partial
from collections import defaultdict
from zipfile import ZipFile

import numpy as np
from PIL import Image

import torch
from torchvision.transforms import ToTensor

from torchnet.dataset import ListDataset, TransformDataset
from torchnet.transform import compose

from protonets.data.base import convert_dict, CudaTransform, EpisodicBatchSampler, SequentialBatchSampler
from protonets.data.omniglot import scale_image


MINIIMAGENET_DATA_DIR  = os.path.join(os.path.dirname(__file__), '../../data/miniImagenet')
MINIIMAGENET_CACHE = { }


def load_image_path(key, out_field, d):
    with ZipFile(os.path.join(MINIIMAGENET_DATA_DIR, 'images.zip'), 'r') as arch:
        fname = d[key]
        with arch.open(fname) as img:
            d[out_field] = Image.open(img)
    return d


def convert_tensor(key, d):
    d[key] = 1.0 - torch.from_numpy(np.array(d[key], np.float32, copy=False)).transpose(0, 1).contiguous().view(1, d[key].size[0], d[key].size[1])
    return d


def load_class_images(class_index, d):
    if d['class'] not in MINIIMAGENET_CACHE:

        image_dir = os.path.join('images')

        class_images = [os.path.join(image_dir, '%s.jpg' % img)
                        for img in class_index[d['class']]]
        if len(class_images) == 0:
            raise Exception("No images found for class %s." % d['class'])

        image_ds = TransformDataset(ListDataset(class_images),
                                    compose([partial(convert_dict, 'file_name'),
                                             partial(load_image_path, 'file_name', 'data'),
                                             #partial(rotate_image, 'data', float(rot[3:])),
                                             partial(scale_image, 'data', 84, 84),
                                             partial(convert_tensor, 'data')]))

        loader = torch.utils.data.DataLoader(image_ds, batch_size=len(image_ds), shuffle=False)

        for sample in loader:
            MINIIMAGENET_CACHE[d['class']] = sample['data']
            break # only need one sample because batch size equal to dataset length

    return { 'class': d['class'], 'data': MINIIMAGENET_CACHE[d['class']] }

def extract_episode(n_support, n_query, d):
    # data: N x C x H x W
    n_examples = d['data'].size(0)

    if n_query == -1:
        n_query = n_examples - n_support

    example_inds = torch.randperm(n_examples)[:(n_support+n_query)]
    support_inds = example_inds[:n_support]
    query_inds = example_inds[n_support:]

    xs = d['data'][support_inds]
    xq = d['data'][query_inds]

    return {
        'class': d['class'],
        'xs': xs,
        'xq': xq
    }

def load(opt, splits):
    split_dir = os.path.join(MINIIMAGENET_DATA_DIR, 'splits', opt['data.split'])

    ret = { }
    for split in splits:
        if split in ['val', 'test'] and opt['data.test_way'] != 0:
            n_way = opt['data.test_way']
        else:
            n_way = opt['data.way']

        if split in ['val', 'test'] and opt['data.test_shot'] != 0:
            n_support = opt['data.test_shot']
        else:
            n_support = opt['data.shot']

        if split in ['val', 'test'] and opt['data.test_query'] != 0:
            n_query = opt['data.test_query']
        else:
            n_query = opt['data.query']

        if split in ['val', 'test']:
            n_episodes = opt['data.test_episodes']
        else:
            n_episodes = opt['data.train_episodes']

        class_index = defaultdict(list)
        with open(os.path.join(split_dir, "{:s}.csv".format(split)), 'r') as f:
            f.readline()
            for image_class in f.readlines():
                image, class_name = image_class.split(',')
                class_name = class_name.rstrip('\n')
                class_index[class_name].append(image)
        class_names = class_index.keys()

        transforms = [partial(convert_dict, 'class'),
                      partial(load_class_images, class_index),
                      partial(extract_episode, n_support, n_query)]
        if opt['data.cuda']:
            transforms.append(CudaTransform())

        transforms = compose(transforms)


        ds = TransformDataset(ListDataset(class_names), transforms)

        if opt['data.sequential']:
            sampler = SequentialBatchSampler(len(ds))
        else:
            sampler = EpisodicBatchSampler(len(ds), n_way, n_episodes)

        # use num_workers=0, otherwise may receive duplicate episodes
        ret[split] = torch.utils.data.DataLoader(ds, batch_sampler=sampler, num_workers=0)

    return ret
