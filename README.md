# What's this
Implementation of PReLUNet by chainer  

# Dependencies

    git clone https://github.com/nutszebra/prelu_net.git
    cd prelu_net
    git submodule init
    git submodule update

# How to run
    python main.py -g 0

# Details about my implementation
All hyperparameters and network architecture are the same as in [[1]][Paper] except for some parts.

* Data augmentation  
Train: Pictures are randomly resized in the range of [256, 512], then 224x224 patches are extracted randomly and are normalized locally. Horizontal flipping is applied with 0.5 probability.  
Test: Pictures are resized to 384x384, then they are normalized locally. Single image test is used to calculate total accuracy.  

* SPP net
Instead of spp, I use global average pooling.

* Learning rate schedule
Learning rate is divided by 10 at [150, 225] epoch. The total number of epochs is 300.

# Cifar10 result
| network                                                   | total accuracy (%) |
|:----------------------------------------------------------|-------------------:|
| my implementation(model A)                                | 94.98              |

<img src="https://github.com/nutszebra/prelu_net/blob/master/img/loss.jpg" alt="loss" title="loss">
<img src="https://github.com/nutszebra/prelu_net/blob/master/img/accuracy.jpg" alt="total accuracy" title="total accuracy">

# References
Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification [[1]][Paper]  

[paper]: https://arxiv.org/abs/1502.01852 "Paper"
