# Approximate-HetConv

## Introduction
This is an approximate but simple pytorch implementation of [HetConv: Heterogeneous Kernel-Based Convolutions for Deep CNNs](https://arxiv.org/abs/1903.04120).
The main purpose is to reduce the number of FLOPs and parameters without sacrificing speed.

## Experimentation
### Results for VGG-16 on CIFAR-10 after 200 epochs
|    Model     |  FLOPs  | Params |  Acc% |
|:------------:|:-------:|:------:|:-----:|
| vgg16_bn_P1  | 313.47M | 14.99M | 93.88 |
| vgg16_bn_P2  | 192.36M |  9.27M | 94.09 |
| vgg16_bn_P4  | 114.50M |  5.59M | 93.74 |
| vgg16_bn_P8  |  75.57M |  3.75M | 93.62 |
| vgg16_bn_P16 |  56.11M |  2.83M | 93.53 |
| vgg16_bn_P32 |  46.38M |  2.37M | 93.06 |
| vgg16_bn_P64 |  41.51M |  2.14M | 92.69 |


