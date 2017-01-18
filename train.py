#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
from chainer.dataset import convert
from chainer.datasets import cifar
from chainer import serializers
from chainer import training
from chainer.training import extensions

import cmd_options
from dataset import PreprocessedDataset
from densenet import DenseNet
from evaluator import Evaluator
from graph import create_fig
from updater import StandardUpdater


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

    images = convert.concat_examples(train)[0]
    mean = images.mean(axis=(0, 2, 3))
    std = images.std(axis=(0, 2, 3))

    train = PreprocessedDataset(train, mean, std, random=args.augment)
    test = PreprocessedDataset(test, mean, std)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(
        test, args.batchsize / args.split_size, repeat=False, shuffle=False)

    model = chainer.links.Classifier(DenseNet(
        n_layer, args.growth_rate, n_class, args.drop_ratio, 16, args.block))
    if args.init_model:
        serializers.load_npz(args.init_model, model)
    chainer.cuda.get_device(args.gpu).use()
    model.to_gpu()

    optimizer = chainer.optimizers.NesterovAG(lr=args.lr, momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))

    updater = StandardUpdater(
        train_iter, optimizer, (args.split_size, 'mean'), device=args.gpu)
    trainer = training.Trainer(updater, (300, 'epoch'), out=args.dir)

    val_interval = (1, 'epoch')
    log_interval = (1, 'epoch')

    def lr_shift():  # DenseNet specific!
        if updater.epoch == 150 or updater.epoch == 225:
            optimizer.lr *= 0.1
        return optimizer.lr

    trainer.extend(Evaluator(
        test_iter, model, device=args.gpu), trigger=val_interval)
    trainer.extend(extensions.observe_value(
        'lr', lambda _: lr_shift()), trigger=(1, 'epoch'))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot_object(
        model, 'epoch_{.updater.epoch}.model'), trigger=val_interval)
    trainer.extend(extensions.snapshot_object(
        optimizer, 'epoch_{.updater.epoch}.state'), trigger=val_interval)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.PrintReport([
        'elapsed_time', 'epoch', 'iteration', 'main/loss', 'validation/main/loss',
        'main/accuracy', 'validation/main/accuracy', 'lr',
    ]), trigger=log_interval)
    trainer.extend(extensions.observe_value(
        'graph', lambda _: create_fig(args.dir)),
        trigger=(1, 'epoch'), priority=50)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    trainer.run()


if __name__ == '__main__':
    args = cmd_options.get_arguments()
    main(args)
