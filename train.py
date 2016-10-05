#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
from chainer.datasets import cifar
from chainer import serializers
from chainer import training
from chainer.training import extensions

import time

import cmd_options
from dataset import PreprocessedDataset
from densenet import DenseNet
from graph import create_fig


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

    train = PreprocessedDataset(train, random=args.augment)
    test = PreprocessedDataset(test)

    train_iter = chainer.iterators.MultiprocessIterator(train, args.batchsize)
    test_iter = chainer.iterators.MultiprocessIterator(
        test, args.batchsize, repeat=False, shuffle=False)

    model = chainer.links.Classifier(DenseNet(
        n_layer, args.growth_rate, n_class, args.drop_ratio, 16, args.block))
    if args.init_model:
        serializers.load_npz(args.init_model, model)

    optimizer = chainer.optimizers.MomentumSGD(
        lr=args.lr / len(args.gpus), momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))

    devices = {'main': args.gpus[0]}
    if len(args.gpus) > 1:
        for gid in args.gpus[1:]:
            devices['gpu%d' % gid] = gid
    updater = training.ParallelUpdater(train_iter, optimizer, devices=devices)
    trainer = training.Trainer(updater, (300, 'epoch'), out=args.dir)

    val_interval = (1, 'epoch')
    log_interval = (1, 'epoch')

    eval_model = model.copy()
    eval_model.train = False

    def lr_shift():  # DenseNet specific!
        if updater.epoch == 151 or updater.epoch == 226:
            optimizer.lr *= 0.1
        return optimizer.lr

    trainer.extend(extensions.Evaluator(
        test_iter, eval_model, device=args.gpus[0]), trigger=val_interval)
    trainer.extend(extensions.observe_value(
        'lr', lambda _: lr_shift()), trigger=(1, 'epoch'))
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
    trainer.extend(extensions.observe_value(
        'graph', lambda _: create_fig(args.dir)), trigger=(2, 'epoch'))
    trainer.extend(extensions.ProgressBar(update_interval=10))

    trainer.run()


if __name__ == '__main__':
    args = cmd_options.get_arguments()
    main(args)
