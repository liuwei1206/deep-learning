#author = liuwei

import torch
import os
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

class LRN(nn.Module):
    '''LRN(local response normalization), do the normalization'''
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=True):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average=nn.AvgPool3d(kernel_size=(local_size, 1, 1), 
                                      stride=1, 
                                      padding=(int((local_size-1.0)/2), 0, 0))
        else:
            self.average=nn.AvgPool2d(kernel_size=local_size,
                                      stride=1,
                                      padding=int((local_size-1.0)/2))
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        # a tuple?
        self.features = nn.Sequential(
            #the first layer net
            #3 inputs channel, 96 output channel, filter kernel size is 11*11, stride is 4, padding is 0
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.Relu(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            LRN(local_size=5, alpha=0.0001, beta=0.75),
            
            #the second
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.Relu(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            LRN(local_size=5, alpha=0.0001, beta=0.75)

            #the third
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.Relu(inplace=True)

            #thr fourth
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.Relu(inplace=True)

            #the fifth
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.Relu(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            #the sixth, full connection, input_size, output_size
            nn.Linear(6*6*256, 4096),
            nn.Relu(inplace=True),

            #the seventh
            nn.Linear(4096, 4096),
            nn.Relu(inplace=True),
            nn.Dropout(),

            #the eigth
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256*6*6)
        x = x.self.classifier(x)
        return x
    

