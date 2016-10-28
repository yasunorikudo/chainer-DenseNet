#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def create_fig(out_dir):
    # load data
    data = json.load(open(os.path.join(out_dir, 'log')))
    train_loss = []
    valid_loss = []
    train_error = []
    valid_error = []
    for d in data:
        train_loss.append(d['main/loss'])
        valid_loss.append(d['validation/main/loss'])
        train_error.append(1 - d['main/accuracy'])
        valid_error.append(1 - d['validation/main/accuracy'])
    x = range(1, len(train_loss) + 1)

    # show graph
    fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10, 5), sharex=True)
    axL.plot(x, valid_error, label='valid error')
    axL.plot(x, train_error, label='train error')
    axL.set_title(
        'Error (min. valid error: {0:.3f}%)'.format(min(valid_error) * 100))
    axL.set_xlabel('epochs')
    axL.set_ylabel('error')
    axL.legend(loc='upper right')
    axR.plot(x, valid_loss, label='valid loss')
    axR.plot(x, train_loss, label='train loss')
    axR.set_title('Loss')
    axR.set_xlabel('epochs')
    axR.set_ylabel('loss')
    axR.legend(loc='upper right')
    fig.savefig(os.path.join(out_dir, 'graph.png'))
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', '-o', type=str)
    args = parser.parse_args()

    create_fig(args.out_dir)
    print('Saved fig as \'{}\' !!'.format(
        os.path.join(args.out_dir, 'graph.png')))
