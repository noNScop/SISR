import math
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


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
    

# ESRGAN
# Generator - EDSR with RRDB

class DenseBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential()
        for i in range(1,5):
            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(i * 64, 64, 3, stride=1, padding='same'),
                    nn.LeakyReLU()
                )
            )

        self.blocks.append(nn.Conv2d(5 * 64, 64, 3, stride=1, padding='same'))

    def forward(self, x):
        x1 = self.blocks[0](x)
        x = torch.cat([x, x1], dim=1)
        x2 = self.blocks[1](x)
        x = torch.cat([x, x2], dim=1)
        x3 = self.blocks[2](x)
        x = torch.cat([x, x3], dim=1)
        x4 = self.blocks[3](x)
        x = torch.cat([x, x4], dim=1)
        return self.blocks[4](x)
        
class ESRGAN(nn.Module):
    def __init__(self, n: int):
        """
        Enhanced Deep Residual Network with Residual in Residual Dense Block a.k.a.
        Enhanced Super Resolution Generative Adversarial Networks
        Args:
            n: scaling factor
        """
        super().__init__()
        self.DIV2K_RGB = torch.tensor([0.44882884613943946, 0.43713809810624193, 0.4040371984052683]).view(1, 3, 1, 1).to(device)
        
        self.expand = nn.Sequential(
            nn.Conv2d(3, 64, 9, stride=1, padding='same'),
            nn.PReLU()
        )

        self.residual_blocks = nn.Sequential()
        for _ in range(23):
            self.residual_blocks.append(DenseBlock())

        self.residual_blocks.append(nn.Conv2d(64, 64, 3, stride=1, padding='same'))

        self.upscaling_head = nn.Sequential()
        for _ in range(int(math.log2(n))):
            self.upscaling_head.append(nn.Conv2d(64, 256, 3, stride=1, padding='same'))
            self.upscaling_head.append(nn.PixelShuffle(2))
            self.upscaling_head.append(nn.PReLU())
            
        self.upscaling_head.append(nn.Conv2d(64, 64, 9, stride=1, padding='same'))
        self.upscaling_head.append(nn.Conv2d(64, 3, 9, stride=1, padding='same'))

    def forward(self, x):
        x = self.expand(x-self.DIV2K_RGB)
        xp = x.clone()
        for i in range(23):
            xp = xp + 0.2 * self.residual_blocks[i](xp)

        x = x + 0.2 * self.residual_blocks[23](xp)
        return self.upscaling_head(x) + self.DIV2K_RGB
    
# Discriminator

class ConvBlock(nn.Module):
    def __init__(self, ni: int, nf: int, ks: int, stride: int):
        super().__init__()
        
        self.block = nn.Sequential(
            spectral_norm(nn.Conv2d(ni, nf, ks, stride=stride, padding=1)),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x):
        return self.block(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.expand = nn.Sequential(
            spectral_norm(nn.Conv2d(3, 64, 3, stride=1, padding=1)),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.body = nn.Sequential(
            ConvBlock(64, 64, 3, 2),
            ConvBlock(64, 128, 3, 1),
            ConvBlock(128, 128, 3, 2),
            ConvBlock(128, 256, 3, 1),
            ConvBlock(256, 256, 3, 2),
            ConvBlock(256, 512, 3, 1),
            ConvBlock(512, 512, 3, 2)
        )

        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        self.head = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(1024, 1)
        )

        self.model = nn.Sequential(
            self.expand,
            self.body,
            self.avgpool,
            self.head   
        )

    def forward(self, x):
        return self.model(x)

class RaD(nn.Module):
    """
    Relativistic average Discriminator
    """
    def __init__(self):
        super().__init__()
        self.discriminator = Discriminator()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        return self.sigmoid(self.discriminator(x1) - self.discriminator(x2).mean(dim=0, keepdim=True))
