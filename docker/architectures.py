import math
import torch
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RCAB(nn.Module):
    """
    Residual Channel Attention Block
    """
    def __init__(self):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding='same')
        )

        self.attention_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(64, 64//16, 1),
            nn.ReLU(),
            nn.Conv2d(64//16, 64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        Xgb = self.block1(x)
        return x + Xgb * self.attention_block(Xgb)
    
class RG(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential()

        for _ in range(20):
            self.block.append(RCAB())

        self.block.append(nn.Conv2d(64, 64, 3, padding='same'))

    def forward(self, x):
        return x + self.block(x)
    
class RIR(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential()

        for _ in range(10):
            self.block.append(RG())

        self.block.append(nn.Conv2d(64, 64, 3, padding='same'))

    def forward(self, x):
        return x + self.block(x)

class RCAN(nn.Module):
    def __init__(self, n):
        """
        Args:
            n: scaling factor
        """
        super().__init__()
        self.DIV2K_RGB = torch.tensor([0.44882884613943946, 0.43713809810624193, 0.4040371984052683]).view(1, 3, 1, 1).to(device)

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding='same'),
            RIR()
        )

        self.upscaling_head = nn.Sequential()
        for _ in range(int(math.log2(n))):
            self.upscaling_head.append(nn.Conv2d(64, 4*64, 3, padding='same'))
            self.upscaling_head.append(nn.PixelShuffle(2))
            self.upscaling_head.append(nn.PReLU())
            
        self.upscaling_head.append(nn.Conv2d(64, 3, 3, padding='same'))

    def forward(self, x):
        return self.upscaling_head(self.feature_extractor(x-self.DIV2K_RGB)) + self.DIV2K_RGB