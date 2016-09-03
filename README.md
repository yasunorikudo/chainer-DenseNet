Densely Connected Convolutional Network implementation by Chainer
========

Implementation by Chainer. Original paper is [Densely Connected Convolutional Network](https://arxiv.org/abs/1608.06993).

This repository includes network definition scripts only.

# Requirements

- [Chainer 1.5+](https://github.com/pfnet/chainer) (Neural network framework)


# Training for cifar-10

Use [mitmul's repository](https://github.com/mitmul/chainer-cifar10/tree/34aa43ff1f1bc559b02bdfc7953747aa84048612).

Run:
```
git clone https://github.com/mitmul/chainer-cifar10.git
cd chainer-cifar10
git checkout 34aa43ff1f1bc559b02bdfc7953747aa84048612
cp ../densenet.py models
bash download.sh
python train.py --model models/densenet.py --batchsize 64 --lr 0.1 --epoch 300 --lr_decay_freq 120
```
