Densely Connected Convolutional Network implementation by Chainer
========

Implementation by Chainer. Original paper is [Densely Connected Convolutional Network](https://arxiv.org/abs/1608.06993).

# Requirements

- [Chainer 1.15.0+](https://github.com/pfnet/chainer) (Neural network framework)

# Start training
For example, run,

```
python train.py --gpus 0 --batchsize 32 --dataset cifar10 --lr 0.1 --depth 100 --growth_rate 12
```

## Show possible options
```
python train.py --help
```


# Sample results

- Cifar-10 (batchsize=32, depth=12, growth_rate=12)

![](https://raw.githubusercontent.com/yasunorikudo/chainer-DenseNet/images/cifar10.png)

- Cifar-100 (batchsize=32, depth=12, growth_rate=12)

![](https://raw.githubusercontent.com/yasunorikudo/chainer-DenseNet/images/cifar100.png)
