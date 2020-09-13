import torch.nn as nn
import torch


class DualConv(nn.Module):
    
    def __init__(self, input_channels, output_channels):
        
        super().__init__()
        
        self.dual_conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        
        return self.dual_conv(x)

class DownConv(nn.Module):
    
    def __init__(self, input_channels, output_channels):
        
        super().__init__()
        
        self.down_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DualConv(input_channels, output_channels)
        )
    
    def forward(self, x):
        
        return self.down_conv(x)

class UpConv(nn.Module):
    
    def __init__(self, input_channels, output_channels):
        
        super().__init__()
        
        self.up_conv = nn.ConvTranspose2d(input_channels, output_channels, kernel_size=2, stride=2)
        self.conv = DualConv(input_channels, output_channels)
    
    def forward(self, x1, x2):
        
        x1 = self.up_conv(x1)
        
        y_pad = x2.size()[2] - x1.size()[2]
        x_pad = x2.size()[3] - x1.size()[3]
        
        x1 = nn.functional.pad(x1, [x_pad // 2, x_pad - x_pad // 2,
                                    y_pad // 2, y_pad - y_pad // 2])
        
        x = torch.cat([x2, x1], dim = 1)
        
        return self.conv(x)

class OutputConv(nn.Module):
    
    def __init__(self, input_channels, output_channels):
        
        super().__init__()
        
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
        
        return self.conv(x)

