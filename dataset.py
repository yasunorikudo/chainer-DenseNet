#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import numpy as np
import random
from six import moves


class PreprocessedDataset(chainer.dataset.DatasetMixin):
    def __init__(self, pairs, mean, std, random=False):
        self._pairs = pairs
        self._mean = mean
        self._std = std
        self._random = random

    def __len__(self):
        return len(self._pairs)

    def get_example(self, i):
        image, label = self._pairs[i]

        # load label data
        t = np.array(label, dtype=np.int32)

        # normalize data
        x = np.empty_like(image)
        for i in moves.range(3):
            x[i] = image[i] - self._mean[i]
            x[i] /= self._std[i]

        # data augmentation
        if self._random:
            # random crop
            pad_x = np.zeros((3, 40, 40), dtype=np.float32)
            pad_x[:, 4:36, 4:36] = x
            top = random.randint(0, 8)
            left = random.randint(0, 8)
            x = pad_x[:, top:top+32, left:left+32]

            # horizontal flip
            if random.randint(0, 1):
                x = x[:, :, ::-1]

        return x, t
