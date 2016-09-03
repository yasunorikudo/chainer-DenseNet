#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import chainer
import chainer.functions as F
import chainer.links as L


class DenseBlock(chainer.Chain):
    def __init__(self, in_ch, out_ch, layer):
        self.layer = layer
        w = math.sqrt(2)
        super(DenseBlock, self).__init__()
        for i in range(self.layer):
            self.add_link('bn%d' % (i + 1),
                L.BatchNormalization(in_ch + i * out_ch))
            self.add_link('conv%d' % (i + 1),
                L.Convolution2D(in_ch + i * out_ch, out_ch, 3, 1, 1, w))

    def __call__(self, x, train):
        hs = [x,]
        for i in range(1, self.layer + 1):
            h = F.relu(self['bn%d' % i](F.concat(hs), test=not train))
            h = self['conv%d' % i](h)
            hs.append(h)
        return F.concat(hs)


class Transition(chainer.Chain):
    def __init__(self, in_ch):
        w = math.sqrt(2)
        super(Transition, self).__init__(
            bn=L.BatchNormalization(in_ch),
            conv=L.Convolution2D(in_ch, in_ch, 1, 1, 0, w))

    def __call__(self, x, train):
        h = F.relu(self.bn(x, test=not train))
        h = F.average_pooling_2d(self.conv(h), 2)
        return h


class DenseNet(chainer.Chain):

    insize = 32

    def __init__(self, layer=12, out_ch=12):

        '''
        layer:  Number of convolution layers in one dense block.
                If layer=12, the network is made out of 40 (12*3+4) layers.
                If layer=32, the network is made out of 100 (32*3+4) layers.
        out_ch: Number of output feature maps of each convolution layer in
                dense blocks, which is difined as growth rate(k) in the paper.
        '''

        w = math.sqrt(2)
        in_chs = range(16, 16 + 4 * layer * out_ch, layer * out_ch)
        super(DenseNet, self).__init__(
            conv1=L.Convolution2D(3, in_chs[0], 3, 1, 1, w),
            dense2=DenseBlock(in_chs[0], out_ch, layer),
            trans2=Transition(in_chs[1]),
            dense3=DenseBlock(in_chs[1], out_ch, layer),
            trans3=Transition(in_chs[2]),
            dense4=DenseBlock(in_chs[2], out_ch, layer),
            bn4=L.BatchNormalization(in_chs[3]),
            fc5=L.Linear(in_chs[3], 10))
        self.train = True

    def clear(self):
        self.loss = None
        self.accuracy = None

    def __call__(self, x, t):
        self.clear()
        h = self.conv1(x)
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
