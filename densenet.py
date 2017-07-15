#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
from chainer import initializers
import chainer.links as L

import numpy as np
from six import moves


class DenseBlock(chainer.Chain):
    def __init__(self, in_ch, growth_rate, n_layer):
        self.n_layer = n_layer
        initialW = initializers.HeNormal()
        super(DenseBlock, self).__init__()
        for i in moves.range(self.n_layer):
            self.add_link('bn{}'.format(i + 1),
                          L.BatchNormalization(in_ch + i * growth_rate))
            self.add_link('conv{}'.format(i + 1),
                          L.Convolution2D(in_ch + i * growth_rate, growth_rate,
                                          3, 1, 1, initialW=initialW))

    def __call__(self, x, dropout_ratio):
        for i in moves.range(1, self.n_layer + 1):
            h = F.relu(self['bn{}'.format(i)](x))
            h = F.dropout(self['conv{}'.format(i)](h), dropout_ratio)
            x = F.concat((x, h))
        return x


class Transition(chainer.Chain):
    def __init__(self, in_ch):
        initialW = initializers.HeNormal()
        super(Transition, self).__init__()
        with self.init_scope():
            self.bn = L.BatchNormalization(in_ch)
            self.conv = L.Convolution2D(in_ch, in_ch, 1, initialW=initialW)

    def __call__(self, x, dropout_ratio):
        h = F.relu(self.bn(x))
        h = F.dropout(self.conv(h), dropout_ratio)
        h = F.average_pooling_2d(h, 2)
        return h


class DenseNet(chainer.Chain):
    def __init__(self, n_layer=12, growth_rate=12,
                 n_class=10, dropout_ratio=0.2, in_ch=16, block=3):
        """DenseNet definition.

        Args:
            n_layer: Number of convolution layers in one dense block.
                If n_layer=12, the network is made out of 40 (12*3+4) layers.
                If n_layer=32, the network is made out of 100 (32*3+4) layers.
            growth_rate: Number of output feature maps of each convolution
                layer in dense blocks, which is difined as k in the paper.
            n_class: Output class.
            dropout_ratio: Dropout ratio.
            in_ch: Number of output feature maps of first convolution layer.
            block: Number of dense block.

        """
        initialW = initializers.HeNormal()
        in_chs = [in_ch + n_layer * growth_rate * i
                  for i in moves.range(block + 1)]
        super(DenseNet, self).__init__()
        self.add_link(
            'conv1', L.Convolution2D(3, in_ch, 3, 1, 1, initialW=initialW))
        for i in moves.range(block):
            self.add_link('dense{}'.format(i + 2),
                          DenseBlock(in_chs[i], growth_rate, n_layer))
            if not i == block - 1:
                self.add_link('trans{}'.format(i + 2), Transition(in_chs[i + 1]))
        self.add_link(
            'bn{}'.format(block + 1), L.BatchNormalization(in_chs[block]))
        self.add_link('fc{}'.format(block + 2), L.Linear(in_chs[block], n_class))
        self.dropout_ratio = dropout_ratio
        self.block = block

    def __call__(self, x):
        h = self.conv1(x)
        for i in moves.range(2, self.block + 2):
            h = self['dense{}'.format(i)](h, self.dropout_ratio)
            if not i == self.block + 1:
                h = self['trans{}'.format(i)](h, self.dropout_ratio)
        h = F.relu(self['bn{}'.format(self.block + 1)](h))
        h = F.average_pooling_2d(h, h.data.shape[2])
        h = self['fc{}'.format(self.block + 2)](h)
        return h
