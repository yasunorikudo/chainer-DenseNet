#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
from chainer.datasets import cifar
from chainer import serializers
from chainer import training
from chainer.training import extensions

import math
import time

import cmd_options
from densenet import DenseNet


def main(args):

    assert((args.depth - args.block - 1) % args.block == 0)
    n_layer = (args.depth - args.block - 1) / args.block
    if args.dataset == 'cifar10':
        train, test = cifar.get_cifar10()
        n_class = 10
    elif args.dataset == 'cifar100':
        train, test = cifar.get_cifar100()
        n_class = 100
    elif args.dataset == 'SVHN':
        raise NotImplementedError()

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(
        test, args.batchsize, repeat=False, shuffle=False)

    model = chainer.links.Classifier(
        DenseNet(n_layer, args.growth_rate, n_class, args.drop_ratio))
    if args.init:
        serializers.load_npz(args.init, model)

    optimizer = chainer.optimizers.MomentumSGD(
        lr=args.lr / len(args.gpus), momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))

    devices = {'main': args.gpus[0]}
    if len(args.gpus) > 2:
        for gid in args.gpus[1:]:
            devices['gpu%d' % gid] = gid
    updater = training.ParallelUpdater(train_iter, optimizer, devices=devices)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.dir)

    val_interval = (1, 'epoch')
    log_interval = (1, 'epoch')

    eval_model = model.copy()
    eval_model.train = False

    trainer.extend(extensions.Evaluator(
        test_iter, eval_model, device=args.gpus[0]), trigger=val_interval)
    trainer.extend(extensions.ExponentialShift(
        'lr', args.lr_decay_ratio), trigger=(args.lr_decay_freq, 'epoch'))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot_object(
        model, 'epoch_{.updater.epoch}.model'), trigger=val_interval)
    trainer.extend(extensions.snapshot_object(
        optimizer, 'epoch_{.updater.epoch}.state'), trigger=val_interval)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    start_time = time.time()
    trainer.extend(extensions.observe_value(
        'time', lambda _: time.time() - start_time), trigger=log_interval)
    trainer.extend(extensions.PrintReport([
        'time', 'epoch', 'iteration', 'main/loss', 'validation/main/loss',
        'main/accuracy', 'validation/main/accuracy', 'lr',
    ]), trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    trainer.run()

if __name__ == '__main__':
    args = cmd_options.get_arguments()
    main(args)
