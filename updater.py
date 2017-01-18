#!/usr/bin/env python
# -*- coding: utf-8 -*-

from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module
from chainer import optimizer as optimizer_module
from chainer import training
from chainer import variable

import six


class StandardUpdater(training.StandardUpdater):

    def __init__(self, iterator, optimizer, split_size=1,
                 converter=convert.concat_examples,
                 device=None, loss_func=None):
        super(StandardUpdater, self).__init__(
            iterator=iterator,
            optimizer=optimizer,
            converter=converter,
            device=device,
            loss_func=loss_func,
        )
        self._split_size = split_size

    def update_core(self):
        optimizer = self._optimizers['main']
        model = optimizer.target
        model.cleargrads()
        batch = self._iterators['main'].next()
        batch_size = self.get_iterator('main').batch_size

        for i in six.moves.range(self._split_size):
            in_arrays = self.converter(batch[i::self._split_size], self.device)
            grad_scale = len(in_arrays[0]) / float(batch_size)
            in_vars = tuple(variable.Variable(x) for x in in_arrays)
            loss = model(*in_vars) * grad_scale
            loss.backward()

        optimizer.update()


class ParallelUpdater(training.ParallelUpdater):

    def __init__(self, iterator, optimizer, split_size=1,
                converter=convert.concat_examples, models=None,
                devices=None, loss_func=None):
        super(ParallelUpdater, self).__init__(
            iterator=iterator,
            optimizer=optimizer,
            converter=converter,
            models=models,
            devices=devices,
            loss_func=loss_func,
        )
        self._split_size = split_size

    def update_core(self):
        optimizer = self.get_optimizer('main')
        model_main = optimizer.target
        models_others = {k: v for k, v in self._models.items()
                         if v is not model_main}

        batch = self.get_iterator('main').next()
        batch_size = self.get_iterator('main').batch_size

        for model in six.itervalues(self._models):
            model.cleargrads()

        n = len(self._models) * self._split_size
        for j in six.moves.range(self._split_size):

            in_arrays_list = {}
            for i, key in enumerate(six.iterkeys(self._models)):
                start = i + j * len(self._models)
                in_arrays_list[key] = self.converter(
                    batch[start::n], self._devices[key])

            losses = []
            for model_key, model in six.iteritems(self._models):
                if model_key != 'main':
                    model.cleargrads()

                in_arrays = in_arrays_list[model_key]
                grad_scale = len(in_arrays[0]) / float(batch_size)
                loss_func = self.loss_func or model

                in_vars = tuple(variable.Variable(x) for x in in_arrays)
                losses.append(loss_func(*in_vars) * grad_scale)

            for loss in losses:
                loss.backward()

            for model in six.itervalues(models_others):
                model_main.addgrads(model)

        optimizer.update()

        for model in six.itervalues(models_others):
            model.copyparams(model_main)
