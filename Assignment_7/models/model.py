import torch
import torch.nn as nn
import torch.nn.functional as F

class Model_1(nn.Module):
    """
    Target: Initial baseline model to understand the problem
    - Get the set-up right for basic NN training
    - Set Transforms
    - Set Data Loader
    - Set Basic Working Code
    - Set Basic Training & Test Loop
    AND
    - Reduce params < 8k
    - Use GAP instead of FCN
    - Use batchnorm for efficiency
    
    Result:
    - Parameters: 6,922
    - Best Training Accuracy: 98.83
    - Best Test Accuracy: 98.67
    
    Analysis: Model is overfitting and needs regularisation
    """
    def __init__(self):
        super().__init__()

        #INPUT BLOCK
        self.convblock1 = nn.Sequential(
            nn.Conv2d(1, 8, 3), #inch=1, outch=8, size=26, rf=3, j=1
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
    
        #CONV BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(8,8, 3), #inch=8, outch=8, size=24, rf=5, j=1
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        self.convblock3 = nn.Sequential(
            nn.Conv2d(8,8, 3), #inch=8, inout=8, size=22, rf=7, j=1
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        self.convblock4 = nn.Sequential(
            nn.Conv2d(8,16, 3), #inch=8, inout=16, size=20, rf=9, j=1
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        # TRANSITION BLOCK
        self.convblock5 = nn.Sequential(
            nn.Conv2d(16,8,3), #inch32, outch=8, size=18, rf=11, j=1
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
    
        #CONV BLOCK 2
        self.convblock6 = nn.Sequential(
            nn.Conv2d(8,8,3), #inch=8, outch=8, size=16, rf=13, j=1
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        self.convblock7 = nn.Sequential(
            nn.Conv2d(8,16,3), #inch=8, outch=16, size=12, rf=17, j=1
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        # OUTPUT BLOCK
        self.convblock8 = nn.Sequential(
            nn.Conv2d(16,10,3), #inch=16, outch=10, size=10, rf=19, j=1
        )
        self.gap = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = self.gap(x)
        x = x.view(x.size(0),-1)
        return F.log_softmax(x, dim=1)

class Model_2(nn.Module):
    """
    Target: Make model more regularised
    - lets add dropout to all the layers
    
    Result:
    - Parameters: 6,922
    - Best Training Accuracy: 98.22
    - Best Test Accuracy: 98.65
    
    Analysis: Model is not overfitting anymore but not reaching desired accuracy of 99.4 due to low capacity
    """
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.1

        #INPUT BLOCK
        self.convblock1 = nn.Sequential(
            nn.Conv2d(1, 8, 3),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
    
        #CONV BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(8,8, 3),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        self.convblock3 = nn.Sequential(
            nn.Conv2d(8,8, 3),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        self.convblock4 = nn.Sequential(
            nn.Conv2d(8,16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        # TRANSITION BLOCK
        self.convblock5 = nn.Sequential(
            nn.Conv2d(16,8,3),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
    
        #CONV BLOCK 2
        self.convblock6 = nn.Sequential(
            nn.Conv2d(8,8,3),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        self.convblock7 = nn.Sequential(
            nn.Conv2d(8,16,3),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        # OUTPUT BLOCK
        self.convblock8 = nn.Sequential(
            nn.Conv2d(16,10,3),
            nn.ReLU()
        )
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.dropout(x)
        x = self.convblock2(x)
        x = self.dropout(x)
        x = self.convblock3(x)
        x = self.dropout(x)
        x = self.convblock4(x)
        x = self.dropout(x)
        x = self.convblock5(x)
        x = self.dropout(x)
        x = self.convblock6(x)
        x = self.dropout(x)
        x = self.convblock7(x)
        x = self.dropout(x)
        x = self.convblock8(x)
        x = self.gap(x)
        x = x.view(x.size(0),-1)
        return F.log_softmax(x, dim=1)

# class Model_3(nn.Module):
#     """
#     Target: Improve model capacity and efficiency
#     - Add more params, add pool layer
#     - Add Image augmentation (Rotation and RandomErasing)
#     - Try LR scheduler experiments
    
#     Result:
#     - Parameters: 7,394
#     - Best Training Accuracy: 97.48
#     - Best Test Accuracy: 99.45
    
#     Analysis: Model is able to achieve accuracy of >99.4 consistently over last few epochs
#     LR scheduler parameters helped in consistency of results
#     """
#     def __init__(self):
#         super().__init__()
#         self.dropout_p = 0.1

#         #INPUT BLOCK
#         self.convblock1 = nn.Sequential(
#             nn.Conv2d(1, 8, 3),
#             nn.BatchNorm2d(8),
#             nn.ReLU()
#         )
    
#         # CONV BLOCK 1
#         self.convblock2 = nn.Sequential(
#             nn.Conv2d(8,8, 3),
#             nn.BatchNorm2d(8),
#             nn.ReLU()
#         )
#         self.convblock3 = nn.Sequential(
#             nn.Conv2d(8,8, 3),
#             nn.BatchNorm2d(8),
#             nn.ReLU()
#         )
#         self.convblock4 = nn.Sequential(
#             nn.Conv2d(8,16, 3),
#             nn.BatchNorm2d(16),
#             nn.ReLU()
#         )
#         # TRANSITION BLOCK
#         self.convblock5 = nn.Sequential(
#             nn.Conv2d(16,8,3),
#             nn.BatchNorm2d(8),
#             nn.ReLU()
#         )
    
#         #CONV BLOCK 2
#         self.convblock6 = nn.Sequential(
#             nn.Conv2d(8,8,3),
#             nn.BatchNorm2d(8),
#             nn.ReLU()
#         )
#         self.convblock7 = nn.Sequential(
#             nn.Conv2d(8,16,3),
#             nn.BatchNorm2d(16),
#             nn.ReLU()
#         )
#         self.convblock8 = nn.Sequential(
#             nn.Conv2d(16,16,3),
#             # nn.BatchNorm2d(16),
#             # nn.ReLU()
#         )

#         # OUTPUT BLOCK
#         self.convblock9 = nn.Sequential(
#             nn.Conv2d(16,10,1),
#             # nn.BatchNorm2d(10),
#             # nn.ReLU()
#         )
#         self.gap = nn.AdaptiveAvgPool2d((1,1))
#         self.dropout = nn.Dropout(self.dropout_p)
#         self.pool = nn.MaxPool2d(2,2)

#     def forward(self, x):
#         x = self.convblock1(x) # chin=1, outch=8, size=26, rf=3, j=1
#         # print(f"convblock1: {x.shape}")
#         x = self.dropout(x)
#         x = self.convblock2(x) # chin=8, outch=8, size=24, rf=5, j=1
#         # print(f"convblock2: {x.shape}")
#         x = self.dropout(x)
#         x = self.convblock3(x) # chin=8, outch=8, size=22, rf=7, j=1
#         # print(f"convblock3: {x.shape}")
#         x = self.dropout(x)
#         x = self.convblock4(x) # chin=8, outch=16, size=20, rf=9, j=1
#         # print(f"convblock4: {x.shape}")
#         x = self.dropout(x)
#         x = self.pool(x) # size=10, rf=10, j=2
#         # print(f"pool: {x.shape}")
#         x = self.convblock5(x) # chin=16, outch=8, size=8, rf=14, j=2
#         # print(f"convblock5: {x.shape}")
#         x = self.dropout(x)
#         x = self.convblock6(x) # chin=8, outch=8, size=6, rf=18, j=2
#         # print(f"convblock6: {x.shape}")
#         x = self.dropout(x)
#         x = self.convblock7(x) # chin=8, outch=16, size=4, rf=22, j=2
#         # print(f"convblock7: {x.shape}")
#         x = self.dropout(x)
#         x = self.convblock8(x) # chin=16, outch=16, size=2, rf=26, j=2
#         # print(f"convblock8: {x.shape}")
#         # x = self.convblock9(x) # chin=16, outch=10, size=2, rf=28, j=2
#         # print(f"convblock9: {x.shape}")
#         x = self.gap(x)
#         # print(f"gap: {x.shape}")
#         # 
#         x = x.view(x.size(0),-1)
#         # print(f"view: {x.shape}")
#         return F.log_softmax(x, dim=1) 
    

class Model_3(nn.Module):
    """
    Target: Improve model capacity and efficiency
    - Add more params, add pool layer
    - Add Image augmentation (Rotation and RandomErasing)
    - Try LR scheduler experiments
    
    Result:
    - Parameters: 7,394
    - Best Training Accuracy: 97.48
    - Best Test Accuracy: 99.45
    
    Analysis: Model is able to achieve accuracy of >99.4 consistently over last few epochs
    LR scheduler parameters helped in consistency of results
    """
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.1

        #INPUT BLOCK
        self.convblock1 = nn.Sequential(
            nn.Conv2d(1, 8, 3),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
    
        # CONV BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(8,16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.convblock3 = nn.Sequential(
            nn.Conv2d(16,16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.convblock4 = nn.Sequential(
            nn.Conv2d(16,8, 3),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        # TRANSITION BLOCK
        self.convblock5 = nn.Sequential(
            nn.Conv2d(8,8,3),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
    
        #CONV BLOCK 2
        self.convblock6 = nn.Sequential(
            nn.Conv2d(8,8,3),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        self.convblock7 = nn.Sequential(
            nn.Conv2d(8,16,3),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.convblock8 = nn.Sequential(
            nn.Conv2d(16,16,1),
            # nn.BatchNorm2d(16),
            # nn.ReLU()
        )

        # OUTPUT BLOCK
        self.convblock9 = nn.Sequential(
            nn.Conv2d(16,10,1),
            # nn.BatchNorm2d(10),
            # nn.ReLU()
        )
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(self.dropout_p)
        self.pool = nn.MaxPool2d(2,2)

    def forward(self, x):
        x = self.convblock1(x) # chin=1, outch=8, size=26, rf=3, j=1
        # print(f"convblock1: {x.shape}")
        x = self.dropout(x)
        x = self.convblock2(x) # chin=8, outch=8, size=24, rf=5, j=1
        # print(f"convblock2: {x.shape}")
        x = self.dropout(x)
        x = self.convblock3(x) # chin=8, outch=8, size=22, rf=7, j=1
        # print(f"convblock3: {x.shape}")
        x = self.dropout(x)
        x = self.convblock4(x) # chin=8, outch=16, size=20, rf=9, j=1
        # print(f"convblock4: {x.shape}")
        x = self.dropout(x)
        x = self.pool(x) # size=10, rf=10, j=2
        # print(f"pool: {x.shape}")
        x = self.convblock5(x) # chin=16, outch=8, size=8, rf=14, j=2
        # print(f"convblock5: {x.shape}")
        x = self.dropout(x)
        x = self.convblock6(x) # chin=8, outch=8, size=6, rf=18, j=2
        # print(f"convblock6: {x.shape}")
        x = self.dropout(x)
        x = self.convblock7(x) # chin=8, outch=16, size=4, rf=22, j=2
        # print(f"convblock7: {x.shape}")
        x = self.dropout(x)
        x = self.gap(x)
        x = self.convblock8(x) # chin=16, outch=16, size=2, rf=26, j=2
        # print(f"convblock8: {x.shape}")
        # x = self.convblock9(x) # chin=16, outch=10, size=2, rf=28, j=2
        # print(f"convblock9: {x.shape}")
        
        # print(f"gap: {x.shape}")
        # 
        x = x.view(x.size(0),-1)
        # print(f"view: {x.shape}")
        return F.log_softmax(x, dim=1) 