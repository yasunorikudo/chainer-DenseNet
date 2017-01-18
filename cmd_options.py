#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import time


def create_log(args):
    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] %(message)s',
        filename=os.path.join(args.dir, 'options.txt'), level=logging.DEBUG)
    logging.info(args)


def create_result_dir(dir):
    if not os.path.exists('results'):
        os.mkdir('results')
    if dir:
        result_dir = os.path.join('results', dir)
    else:
        result_dir = os.path.join(
            'results', time.strftime('%Y-%m-%d_%H-%M-%S'))
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    return result_dir


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, default='0',
                        help='Set GPU device numbers with comma saparated. '
                        'Default is 0.')
    parser.add_argument('--augment', type=str, default='t', choices=['t', 'f'],
                        help='Whether to use data augmentation. '
                        'Default is \'t\'')
    parser.add_argument('--batchsize', type=int, default=64,
                        help='Default is 64.')
    parser.add_argument('--split_size', type=int, default=1,
                        help='Split size of batch in order to reduce GPU mem'
                        'ory. Default is 1.')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Initial learning rate. Default is 0.1')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Default is 1e-4.')
    parser.add_argument('--depth', type=int, default=40,
                        help='Depth of DenseNet, which is defined as L in '
                        'the paper. Default is 40.')
    parser.add_argument('--growth_rate', type=int, default=12,
                        help='Growth rate which is defined as k in the paper.'
                        ' Default is 12.')
    parser.add_argument('--drop_ratio', type=float, default=0,
                        help='Dropout ratio. The paper recommends 0 with data'
                        ' augmentation and 0.2 without data augmentation. '
                        'Default is 0')
    parser.add_argument('--block', type=int, default=3,
                        help='Number of dense block. Default is 3.')
    parser.add_argument('--init_model', type=str, default='',
                        help='Path to chainer model to load before trainig.')
    parser.add_argument('--dir', type=str, default='',
                        help='Directory name to save logs.')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'SVHN'],
                        help='Dataset name. Default \'cifar10\'.')
    args = parser.parse_args()

    args.gpus = map(int, args.gpus.split(','))
    args.augment = True if args.augment == 't' else False
    args.dir = create_result_dir(args.dir)

    create_log(args)

    return args
