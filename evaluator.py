#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import six

from chainer import reporter as reporter_module
from chainer.training import extensions
from chainer import variable


class Evaluator(extensions.Evaluator):

    def evaluate(self):
        """Patch implementation as Chain.copy seems not to copy params in
        chainer.links.BatchNormalization.
        """
        iterator = self._iterators['main']
        target = self._targets['main']
        target.predictor.train = False  # additional line
        eval_func = self.eval_func or target

        if self.eval_hook:
            self.eval_hook(self)
        it = copy.copy(iterator)
        summary = reporter_module.DictSummary()

        for batch in it:
            observation = {}
            with reporter_module.report_scope(observation):
                in_arrays = self.converter(batch, self.device)
                if isinstance(in_arrays, tuple):
                    in_vars = tuple(variable.Variable(x, volatile='on')
                                    for x in in_arrays)
                    eval_func(*in_vars)
                elif isinstance(in_arrays, dict):
                    in_vars = {key: variable.Variable(x, volatile='on')
                               for key, x in six.iteritems(in_arrays)}
                    eval_func(**in_vars)
                else:
                    in_var = variable.Variable(in_arrays, volatile='on')
                    eval_func(in_var)

            summary.add(observation)
        target.predictor.train = True  # additional line

        return summary.compute_mean()
