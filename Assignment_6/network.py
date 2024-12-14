import torch
import torch.nn as nn
import torch.nn.functional as F

dropout_value = 0.1

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        #INPUT BLOCK
        self.convblock1 = nn.Sequential(
            nn.Conv2d(1, 8, 3), #inch=1, outch=8, size=26, rf=3, j=1
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        )
    
        #CONV BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(8,16, 3), #inch=8, outch=16, size=24, rf=5, j=1
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        )
        self.convblock3 = nn.Sequential(
            nn.Conv2d(16,16, 3), #inch=16, inout=16, size=22, rf=7, j=1
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        )

        self.convblock31 = nn.Sequential(
            nn.Conv2d(16,16, 3), #inch=16, inout=16, size=20, rf=9, j=1
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        )
        # TRANSITION BLOCK
        #POOL LAYER
        self.convblock4 = nn.Sequential(
            nn.Conv2d(16,8,3), #inch16, outch=8, size=18, rf=11, j=1
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 9, rf=12, j=2
    
        #CONV BLOCK 2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(8,16,3), #inch=8, outch=16, size=7, rf=16, j=2
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        )
        self.convblock6 = nn.Sequential(
            nn.Conv2d(16,16,3), #inch=16, outch=16, size=5, rf=20, j=2
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        )

        # OUTPUT BLOCK
        self.convblock7 = nn.Sequential(
            nn.Conv2d(16,32,3), #inch=16, outch=32, size=3, rf=24, j=2
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        )
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)
        ) # output_size = 5 #inch=32, outch=10, size=1, rf=28, j=2
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock31(x)
        x = self.convblock4(x)
        x = self.pool1(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1) 