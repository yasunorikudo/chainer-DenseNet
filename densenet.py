#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L

import math
import numpy as np


class DenseBlock(chainer.Chain):
    def __init__(self, in_ch, growth_rate, layer):
        self.layer = layer
        super(DenseBlock, self).__init__()
        for i in range(self.layer):
            W = np.random.randn(growth_rate, in_ch + i * growth_rate, \
                3, 3).astype(np.float32) * math.sqrt(2. / 9 / growth_rate)
            self.add_link('bn%d' % (i + 1),
                          L.BatchNormalization(in_ch + i * growth_rate))
            self.add_link('conv%d' % (i + 1),
                          L.Convolution2D(in_ch + i * growth_rate,
                                          growth_rate, 3, 1, 1, initialW=W))

    def __call__(self, x, train):
        for i in range(1, self.layer + 1):
            h = F.relu(self['bn%d' % i](x, test=not train))
            h = F.dropout(self['conv%d' % i](h), 0.2, train=train)
            x = F.concat((x, h))
        return x


class Transition(chainer.Chain):
    def __init__(self, in_ch):
        W = np.random.randn(in_ch, in_ch, 1, 1).astype(np.float32) \
            * math.sqrt(2. / in_ch)
        super(Transition, self).__init__(
            bn=L.BatchNormalization(in_ch),
            conv=L.Convolution2D(in_ch, in_ch, 1, initialW=W))

    def __call__(self, x, train):
        h = F.relu(self.bn(x, test=not train))
        h = F.dropout(self.conv(h), 0.2, train=train)
        h = F.average_pooling_2d(h, 2)
        return h


class DenseNet(chainer.Chain):

    insize = 32

    def __init__(self, layer=12, growth_rate=12):
        """

        layer: Number of convolution layers in one dense block.
            If layer=12, the network is made out of 40 (12*3+4) layers.
            If layer=32, the network is made out of 100 (32*3+4) layers.
        growth_rate: Number of output feature maps of each convolution layer
            in dense blocks, which is difined as k in the paper.

        """
        in_chs = range(16, 16 + 4 * layer * growth_rate, layer * growth_rate)
        W = np.random.randn(in_chs[0], 3, 3, 3).astype(np.float32) * \
            math.sqrt(2. / 9 / in_chs[0])
        super(DenseNet, self).__init__(
            conv1=L.Convolution2D(3, in_chs[0], 3, 1, 1, initialW=W),
            dense2=DenseBlock(in_chs[0], growth_rate, layer),
            trans2=Transition(in_chs[1]),
            dense3=DenseBlock(in_chs[1], growth_rate, layer),
            trans3=Transition(in_chs[2]),
            dense4=DenseBlock(in_chs[2], growth_rate, layer),
            bn4=L.BatchNormalization(in_chs[3]),
            fc5=L.Linear(in_chs[3], 10))
        self.train = True

    def clear(self):
        self.loss = None
        self.accuracy = None

    def __call__(self, x, t):
        self.clear()
        h = F.dropout(self.conv1(x), 0.2, train=self.train)
        h = self.trans2(self.dense2(h, self.train), self.train)
        h = self.trans3(self.dense3(h, self.train), self.train)
        h = F.relu(self.bn4(self.dense4(h, self.train)))
        h = F.average_pooling_2d(h, h.data.shape[2])
        h = self.fc5(h)

        if self.train:
            self.loss = F.softmax_cross_entropy(h, t)
            self.accuracy = F.accuracy(h, t)
            return self.loss
        else:
            return h

model = DenseNet()
