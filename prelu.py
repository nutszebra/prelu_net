import six
import numpy as np
import functools
import chainer.links as L
import chainer.functions as F
from collections import defaultdict
import nutszebra_chainer


class Conv_PReLU(nutszebra_chainer.Model):

    def __init__(self, in_channel, out_channel, filiter=3, stride=1, pad=1, init=0.25):
        super(Conv_PReLU, self).__init__()
        modules = []
        modules += [('conv', L.Convolution2D(in_channel, out_channel, filiter, stride, pad))]
        modules += [('conv_prelu', L.PReLU(out_channel, init=init))]
        # register layers
        [self.add_link(*link) for link in modules]
        self.modules = modules
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.filiter = filiter
        self.stride = stride
        self.pad = pad
        self.init = init

    @staticmethod
    def _count_conv_parameters(conv):
        return functools.reduce(lambda a, b: a * b, conv.W.data.shape)

    def count_parameters(self):
        return self._count_conv_parameters(self.conv)

    def weight_initialization(self):
        self.conv.W.data = self.weight_relu_initialization(self.conv)
        self.conv.b.data = self.bias_initialization(self.conv, constant=0)

    def __call__(self, x):
        return self.conv_prelu(self.conv(x))


class Triple_FC(nutszebra_chainer.Model):

    def __init__(self, in_channel, out_channels=(4096, 4096, 10), init=0.25, dropout_ratio=0.5):
        self.in_channel = in_channel
        self.out_channels = out_channels
        self.droput_ratio = dropout_ratio
        super(Triple_FC, self).__init__()
        modules = []
        for i, out_channel in enumerate(out_channels):
            modules += [('fc{}'.format(i), L.Linear(in_channel, out_channel))]
            modules += [('prelu{}'.format(i), L.PReLU(out_channel, init=init))]
            in_channel = out_channel
        # register layers
        [self.add_link(*link) for link in modules]
        self.modules = modules

    @staticmethod
    def _count_fc_parameters(fc):
        return functools.reduce(lambda a, b: a * b, fc.W.data.shape)

    def count_parameters(self):
        count = 0
        for name, link in self.modules:
            if 'fc' in name:
                count += self._count_fc_parameters(link)
        return count

    def weight_initialization(self):
        for name, link in self.modules:
            if 'fc' in name:
                self[name].W.data = self.weight_relu_initialization(self[name])
                self[name].b.data = self.bias_initialization(self[name], constant=0)

    def __call__(self, x):
        for i in six.moves.range(len(self.out_channels)):
            if i <= (len(self.out_channels) - 1):
                x = F.dropout(x, ratio=self.droput_ratio)
            x = self['fc{}'.format(i)](x)
            x = self['prelu{}'.format(i)](x)
        return x


class PReLUNet(nutszebra_chainer.Model):

    def __init__(self, category_num, block_num=(5, 5, 5), out_channels=(256, 512, 512), init=0.25):
        self.category_num = category_num
        self.block_num = block_num
        self.out_channels = out_channels
        self.init = init
        super(PReLUNet, self).__init__()
        modules = []
        modules += [('conv1', Conv_PReLU(3, 96, 7, 2, 3, init))]
        in_channel = 96
        for i, n in enumerate(block_num):
            for ii in six.moves.range(n):
                out_channel = out_channels[i]
                modules += [('prelu_block{}_{}'.format(i, ii), Conv_PReLU(in_channel, out_channel, init=init))]
                in_channel = out_channel
        modules += [('fc', Triple_FC(out_channels[-1], out_channels=(4096, 4096, category_num), init=init))]
        # register layers
        [self.add_link(*link) for link in modules]
        self.modules = modules
        self.name = 'PReLUNet_{}_{}_{}_{}'.format(category_num, block_num, out_channels, init)

    def count_parameters(self):
        count = 0
        for name, link in self.modules:
            count += link.count_parameters()
        return count

    def weight_initialization(self):
        for name, link in self.modules:
            link.weight_initialization()

    def __call__(self, x, train=True):
        h = self.conv1(x)
        for i, n in enumerate(self.block_num):
            h = F.max_pooling_2d(h, ksize=(2, 2), stride=(2, 2), pad=(0, 0))
            for ii in six.moves.range(n):
                h = self['prelu_block{}_{}'.format(i, ii)](h)
        batch, channels, height, width = h.data.shape
        h = F.reshape(F.average_pooling_2d(h, (height, width)), (batch, channels, 1, 1))
        return self.fc(h)

    def calc_loss(self, y, t):
        loss = F.softmax_cross_entropy(y, t)
        return loss

    def accuracy(self, y, t, xp=np):
        y.to_cpu()
        t.to_cpu()
        indices = np.where((t.data == np.argmax(y.data, axis=1)) == True)[0]
        accuracy = defaultdict(int)
        for i in indices:
            accuracy[t.data[i]] += 1
        indices = np.where((t.data == np.argmax(y.data, axis=1)) == False)[0]
        false_accuracy = defaultdict(int)
        false_y = np.argmax(y.data, axis=1)
        for i in indices:
            false_accuracy[(t.data[i], false_y[i])] += 1
        return accuracy, false_accuracy
