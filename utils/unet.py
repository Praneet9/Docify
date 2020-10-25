import torch.nn as nn
from utils.nn_block import DualConv, DownConv, UpConv, OutputConv


class UNet(nn.Module):
    
    def __init__(self, n_channels, n_classes):
        
        super().__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        self.inp = DualConv(n_channels, 64)
        
        self.down_conv_1 = DownConv(64, 128)
        self.down_conv_2 = DownConv(128, 256)
        self.down_conv_3 = DownConv(256, 512)
        self.down_conv_4 = DownConv(512, 1024)
        
        self.up_conv_1 = UpConv(1024, 512)
        self.up_conv_2 = UpConv(512, 256)
        self.up_conv_3 = UpConv(256, 128)
        self.up_conv_4 = UpConv(128, 64)
        
        self.op_conv = OutputConv(64, n_classes)
    
    def forward(self, x):
        
        x1 = self.inp(x)
        
        x2 = self.down_conv_1(x1)
        x3 = self.down_conv_2(x2)
        x4 = self.down_conv_3(x3)
        x5 = self.down_conv_4(x4)
        
        x6 = self.up_conv_1(x5, x4)
        x7 = self.up_conv_2(x6, x3)
        x8 = self.up_conv_3(x7, x2)
        x9 = self.up_conv_4(x8, x1)
        
        result = self.op_conv(x9)
        
        return result