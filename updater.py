#!/usr/bin/env python
# -*- coding: utf-8 -*-

from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module
from chainer import optimizer as optimizer_module
from chainer import training
from chainer import variable

import six


class StandardUpdater(training.StandardUpdater):

    def __init__(self, iterator, optimizer, batch_split=(1, 'mean'),
                 converter=convert.concat_examples,
                 device=None, loss_func=None):
        super(StandardUpdater, self).__init__(
            iterator=iterator,
            optimizer=optimizer,
            converter=converter,
            device=device,
            loss_func=loss_func,
        )

        self._split_size = batch_split[0]
        if batch_split[1] == 'mean':
            self._grad_divisor = self._split_size
        elif batch_split[1] == 'sum':
            self._grad_divisor = 1
        else:
            raise Exception('batch_split option is \'mean\' or \'sum\'')

    def update_core(self):
        optimizer = self._optimizers['main']
        model = optimizer.target
        model.cleargrads()
        batch = self._iterators['main'].next()

        for i in six.moves.range(self._split_size):
            in_arrays = self.converter(batch[i::self._split_size], self.device)
            in_vars = tuple(variable.Variable(x) for x in in_arrays)
            loss = model(*in_vars) / self._grad_divisor
            loss.backward()

        optimizer.update()
