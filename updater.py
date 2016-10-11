#!/usr/bin/env python
# -*- coding: utf-8 -*-

from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module
from chainer import optimizer as optimizer_module
from chainer import training
from chainer import variable

import six


class StandardUpdater(training.StandardUpdater):

    def __init__(self, iterator, optimizer, update_freq=1,
                 converter=convert.concat_examples,
                 device=None, loss_func=None):
        if isinstance(iterator, iterator_module.Iterator):
            iterator = {'main': iterator}
        self._iterators = iterator

        if isinstance(optimizer, optimizer_module.Optimizer):
            optimizer = {'main': optimizer}
        self._optimizers = optimizer

        self._update_freq = update_freq
        self.converter = converter
        self.loss_func = loss_func
        self.device = device
        self.iteration = 0

    def update_core(self):
        optimizer = self._optimizers['main']
        model = optimizer.target
        model.cleargrads()

        for _ in six.moves.range(self._update_freq):
            batch = self._iterators['main'].next()
            in_arrays = self.converter(batch, self.device)

            in_vars = tuple(variable.Variable(x) for x in in_arrays)
            loss = model(*in_vars)
            loss.backward()

        optimizer.update()
