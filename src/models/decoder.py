import torch.nn as nn
import numpy as np


class Autoencoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=3, padding_type='reflect'):
        super(Autoencoder, self).__init__()

        use_bias = norm_layer == nn.InstanceNorm2d
        model = [nn.Conv2d(input_nc, ngf, kernel_size=1)]
        model = [#nn.ReflectionPad2d(1),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 4
        # Special case for 9th block of resnet
        #n_downsampling, n_blocks = 0, 0
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2), nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                                     kernel_size=3, stride=2,
                                                     padding=1, output_padding=1, bias=use_bias),
                      norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]
        
        n_upsampling_extra = 1
        for i in range(n_upsampling_extra):  # add upsampling layers
            model += [nn.ConvTranspose2d(ngf, ngf,
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1, bias=use_bias),
                      norm_layer(ngf), nn.ReLU(True)]
            if i == 3:
                model += [nn.Conv2d(ngf, ngf,
                                             kernel_size=3, stride=1, padding=0),
                                             norm_layer(ngf), nn.ReLU(True)]#"""

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, ngf//2, kernel_size=7, padding=0)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf//2, ngf//4, kernel_size=5, padding=0)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf//4, output_nc, kernel_size=5, padding=0)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(output_nc, output_nc, kernel_size=5, padding=0)]

        self.m = nn.Sequential(*model)
    
    def forward(self, x):
        for l in self.m:
            x = l(x)
            #print(x.shape)
        return x