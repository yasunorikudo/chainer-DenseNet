#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L

import numpy as np
from six import moves


class DenseBlock(chainer.Chain):
    def __init__(self, in_ch, growth_rate, n_layer):
        self.n_layer = n_layer
        super(DenseBlock, self).__init__()
        for i in moves.range(self.n_layer):
            self.add_link('bn%d' % (i + 1),
                          L.BatchNormalization(in_ch + i * growth_rate))
            self.add_link('conv%d' % (i + 1),
                          L.Convolution2D(in_ch + i * growth_rate, growth_rate,
                                          3, 1, 1, wscale=np.sqrt(2)))

    def __call__(self, x, dropout_ratio, train):
        for i in moves.range(1, self.n_layer + 1):
            h = F.relu(self['bn%d' % i](x, test=not train))
            h = F.dropout(self['conv%d' % i](h), dropout_ratio, train)
            x = F.concat((x, h))
        return x


class Transition(chainer.Chain):
    def __init__(self, in_ch):
        super(Transition, self).__init__(
            bn=L.BatchNormalization(in_ch),
            conv=L.Convolution2D(in_ch, in_ch, 1, wscale=np.sqrt(2)))

    def __call__(self, x, dropout_ratio, train):
        h = F.relu(self.bn(x, test=not train))
        h = F.dropout(self.conv(h), dropout_ratio, train)
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
        in_chs = [in_ch + n_layer * growth_rate * i
                  for i in moves.range(block + 1)]
        super(DenseNet, self).__init__()
        self.add_link(
            'conv1', L.Convolution2D(3, in_ch, 3, 1, 1, wscale=np.sqrt(2)))
        for i in moves.range(block):
            self.add_link('dense%d' % (i + 2),
                          DenseBlock(in_chs[i], growth_rate, n_layer))
            if not i == block - 1:
                self.add_link('trans%d' % (i + 2), Transition(in_chs[i + 1]))
        self.add_link(
            'bn%d' % (block + 1), L.BatchNormalization(in_chs[block]))
        self.add_link('fc%d' % (block + 2), L.Linear(in_chs[block], n_class))
        self.train = True
        self.dropout_ratio = dropout_ratio
        self.block = block

    def __call__(self, x):
        h = self.conv1(x)
        for i in moves.range(2, self.block + 2):
            h = self['dense%d' % i](h, self.dropout_ratio, self.train)
            if not i == self.block + 1:
                h = self['trans%d' % i](h, self.dropout_ratio, self.train)
        h = F.relu(self['bn%d' % (self.block + 1)](h, test=not self.train))
        h = F.average_pooling_2d(h, h.data.shape[2])
        h = self['fc%d' % (self.block + 2)](h)
        return h
