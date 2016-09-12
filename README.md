Densely Connected Convolutional Network implementation by Chainer
========

Implementation by Chainer. Original paper is [Densely Connected Convolutional Network](https://arxiv.org/abs/1608.06993).

# Requirements

- [Chainer 1.15.0+](https://github.com/pfnet/chainer) (Neural network framework)

# Start training
Run,

```
python train.py --gpus 0 --batchsize 64 --dataset cifar100 --lr 0.1 --depth 100 --growth_rate 12
```

# Sample results

- Cifar-10

![](https://raw.githubusercontent.com/yasunorikudo/chainer-DenseNet/images/cifar10.png)

- Cifar-100

![](https://raw.githubusercontent.com/yasunorikudo/chainer-DenseNet/images/cifar100.png)
