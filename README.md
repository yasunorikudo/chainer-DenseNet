Densely Connected Convolutional Network implementation by Chainer
========

Implementation by Chainer. Original paper is [Densely Connected Convolutional Network](https://arxiv.org/abs/1608.06993).

This repository includes network definition scripts only.

# Requirements

- [Chainer 1.5+](https://github.com/pfnet/chainer) (Neural network framework)


# Training for cifar-10

Use [mitmul's repository](https://github.com/mitmul/chainer-cifar10).

Run:
```
git clone https://github.com/mitmul/chainer-cifar10.git
cp densenet.py chainer-cifar10/models
cd chainer-cifar10
bash download.sh
python train.py --model models/densenet.py --lr 0.1 --batchsize 64
```
