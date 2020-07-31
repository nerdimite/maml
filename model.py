import torch
from torch import nn
import torch.nn.functional as F

def conv_block(in_channels, out_channels):
    '''Convolution Block of 3x3 kernels + batch norm + maxpool of 2x2'''
    
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

def functional_conv_block(x, weights, biases, bn_weights, bn_biases):
    '''Functional version of the conv_block'''
    
    x = F.conv2d(x, weights, biases, padding=1)
    x = F.batch_norm(x, running_mean=None, running_var=None, weight=bn_weights, bias=bn_biases, training=True)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=2, stride=2)
    
    return x

class MAMLClassifier(nn.Module):
    
    def __init__(self, n_way):
        super(MAMLClassifier, self).__init__()
        
        self.conv1 = conv_block(3, 64)
        self.conv2 = conv_block(64, 64)
        self.conv3 = conv_block(64, 64)
        self.conv4 = conv_block(64, 64)
        
        self.head = nn.Linear(64, n_way)
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # Features of shape (batch_size, 64)
        feat = x.view(x.size(0), -1)
        
        # Output
        out = self.head(feat)
        
        return out
    
    def functional_forward(self, x, params):
        '''Functional forward pass given the parameters'''
        
        for block in [1,2,3,4]:
            x = functional_conv_block(x, 
                                      params[f'conv{block}.0.weight'], 
                                      params[f'conv{block}.0.bias'],
                                      params[f'conv{block}.1.weight'],
                                      params[f'conv{block}.1.bias'])
        
        # Features of shape (batch_size, 64)   
        feat = x.view(x.size(0), -1)
        
        # Output
        out = F.linear(feat, params['head.weight'], params['head.bias'])
        
        return out