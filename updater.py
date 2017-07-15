#!/usr/bin/env python
# -*- coding: utf-8 -*-

import multiprocessing
import warnings

import six

from chainer import cuda
from chainer.dataset import convert
from chainer import reporter
from chainer.training import updater, updaters
from chainer import variable

try:
    from cupy.cuda import nccl
    _available = True
except ImportError:
    _available = False

import numpy

from chainer.training.updaters import *

class MultiprocessParallelUpdater(updaters.MultiprocessParallelUpdater):


    def update_core(self):
        self.setup_workers()

        self._send_message(('update', None))
        with cuda.Device(self._devices[0]):
            # For reducing memory
            self._master.cleargrads()

            optimizer = self.get_optimizer('main')
            batch = self.get_iterator('main').next()
            # print(type(self.converter))

            batch = self.converter(batch, self._devices[0])

            loss = _calc_loss(self._master, batch)

            self._master.cleargrads()
            loss.backward()

            # NCCL: reduce grads
            null_stream = cuda.Stream.null
            if self.comm is not None:
                gg = gather_grads(self._master)
                self.comm.reduce(gg.data.ptr, gg.data.ptr, gg.size,
                                 nccl.NCCL_FLOAT, nccl.NCCL_SUM,
                                 0, null_stream.ptr)
                scatter_grads(self._master, gg)
                del gg
            optimizer.update()
            if self.comm is not None:
                gp = gather_params(self._master)
                self.comm.bcast(gp.data.ptr, gp.size, nccl.NCCL_FLOAT,
                                0, null_stream.ptr)
