#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import numpy as np
import random


class PreprocessedDataset(chainer.dataset.DatasetMixin):
    def __init__(self, pairs, mean, random=False):
        self._pairs = pairs
        self._mean = mean
        self._random = random

    def __len__(self):
        return len(self._pairs)

    def get_example(self, i):
        image, label = self._pairs[i]

        # load label data
        label = np.array(label, dtype=np.int32)

        # normalize data
        image -= self._mean.astype(np.float32)
        image /= np.std(image)

        # data augmentation
        if self._random:
            pad_image = np.zeros((3, 40, 40), dtype=np.float32)
            pad_image[:, 4:36, 4:36] = image
            top = random.randint(0, 8)
            left = random.randint(0, 8)
            image = pad_image[:, top:top+32, left:left+32]
            if random.randint(0, 1):
                image = image[:, :, ::-1]

        return image, label
