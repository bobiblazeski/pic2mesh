import torch.nn as nn

class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_ch, ker_size, padd, stride):
        super(ConvBlock,self).__init__()
        self.add_module('conv',nn.Conv2d(in_channel, out_ch, 
                                         kernel_size=ker_size,
                                         stride=stride,
                                         padding=padd)),
        self.add_module('norm',nn.BatchNorm2d(out_ch)),
        self.add_module('LeakyRelu',nn.LeakyReLU(0.2, inplace=True))

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('Norm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0) 