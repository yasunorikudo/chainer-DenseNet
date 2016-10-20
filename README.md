Densely Connected Convolutional Network implementation by Chainer
========

Implementation by Chainer. Original paper is [Densely Connected Convolutional Network](https://arxiv.org/abs/1608.06993).

# Requirements

- python 2.7

- [Chainer 1.15.0+](https://github.com/pfnet/chainer) (Neural network framework)

# Start training
For example, run,

```
python train.py --gpu 0 --batchsize 64 --dataset cifar10 --lr 0.1 --depth 100 --growth_rate 24 --split_size 4
```

## Show possible options
```
python train.py --help
```


# Sample results

- Cifar-10 (batchsize=64, depth=100, growth_rate=24, with data augmentation)

![](https://raw.githubusercontent.com/yasunorikudo/chainer-DenseNet/images/cifar10.png)

[Original paper](https://arxiv.org/abs/1608.06993) reported 3.74% validation error under the same conditions.

- Cifar-100 (batchsize=64, depth=100, growth_rate=24, with data augmentation)

![](https://raw.githubusercontent.com/yasunorikudo/chainer-DenseNet/images/cifar100.png)

[Original paper](https://arxiv.org/abs/1608.06993) reported 19.25% validation error under the same conditions.
