import os
import math
import torch
from PIL import Image
import torch.nn as nn
from datetime import datetime
from torchvision.transforms import v2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BicubicInterpolation:
    def __init__(self, scale: int):
        self.name = "Bicubic"
        self.scale = scale

    def __call__(self, image: str):
        image = Image.open(image).convert("RGB")
        scaled_image = image.resize((self.scale * image.width, self.scale * image.height), Image.Resampling.BICUBIC)
        os.makedirs("output", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        scaled_image.save(f"output/{timestamp}_X{self.scale}.png")





class FSRCNN(nn.Module):
    def __init__(self, n: int, d: int = 56, s: int = 12, m: int = 4):
        """
        Args:
            d: feature dimension
            s: shrinking dimension
            m: mapping layers
            n: scaling factor
        """
        super().__init__()
        self.name = "FSRCNN"
        self.scale = n

        self.transform = v2.Compose([
            v2.PILToTensor(),
            v2.Lambda(lambda x: x/255.0)
        ])
        
        self.model = nn.Sequential(
            self._conv(3, d, 5),
            self._conv(d, s, 1)
        )

        for _ in range(m):
            self.model.append(self._conv(s, s, 3))

        self.model.append(self._conv(s, d, 1))

        # Ensure the output image is exactly n times bigger than the input
        if n <= 9:
            padding = (9 - n + 1) // 2
            output_padding = (9 - n) % 2
        else:
            for i in range(n):
                padding = i - n + 9
                if padding % 2 == 0 and padding >= 0:
                    output_padding = i
                    break
        
        self.model.append(nn.ConvTranspose2d(d, 3, 9, stride=n, padding=padding, output_padding=output_padding))

    def forward(self, image: str):
        image = Image.open(image).convert("RGB")
        inp = self.transform(image).to(device)
        out = self.model(inp).clamp(0.0, 1.0) * 255.0
        img = Image.fromarray(out.permute(1, 2, 0).to(torch.uint8).cpu().numpy())
        os.makedirs("output", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img.save(f"output/{timestamp}_X{self.scale}.png")
        
    def _conv(self, ni, nf, ks):
        return nn.Sequential(
            nn.Conv2d(ni, nf, ks, padding='same'),
            nn.PReLU()
        )
    




class ResBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(64, 64, 3, stride=1, padding='same'),
                nn.BatchNorm2d(64),
                nn.PReLU(),
                nn.Conv2d(64, 64, 3, stride=1, padding='same'),
                nn.BatchNorm2d(64)
            )

        def forward(self, x):
            return x + self.block(x)

class SRResNet(nn.Module):
    def __init__(self, n: int):
        """
        Args:
            n: scaling factor
        """
        super().__init__()
        self.name = "SRResNet"
        self.scale = n

        self.transform = v2.Compose([
            v2.PILToTensor(),
            v2.Lambda(lambda x: x / 255.0)
        ])
        
        self.expand = nn.Sequential(
            nn.Conv2d(3, 64, 9, stride=1, padding='same'),
            nn.PReLU()
        )

        self.residual_blocks = nn.Sequential()
        for _ in range(16):
            self.residual_blocks.append(ResBlock())

        self.residual_blocks.append(nn.Conv2d(64, 64, 3, stride=1, padding='same'))
        self.residual_blocks.append(nn.BatchNorm2d(64))

        self.upscaling_head = nn.Sequential()
        for _ in range(int(math.log2(n))):
            self.upscaling_head.append(nn.Conv2d(64, 256, 3, stride=1, padding='same'))
            self.upscaling_head.append(nn.PixelShuffle(2))
            self.upscaling_head.append(nn.PReLU())
            
        self.upscaling_head.append(nn.Conv2d(64, 3, 9, stride=1, padding='same'))
    
    def forward(self, image: str):
        image = Image.open(image).convert("RGB")
        inp = self.transform(image).to(device)
        x = self.expand(inp[None])
        out = (((self.upscaling_head(self.residual_blocks(x) + x)+1.0)/2.0).clamp(0.0, 1.0) * 255.0).squeeze()
        img = Image.fromarray(out.permute(1, 2, 0).to(torch.uint8).cpu().numpy())
        os.makedirs("output", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img.save(f"output/{timestamp}_X{self.scale}.png")





class SRGAN(nn.Module):
    def __init__(self, n: int):
        """
        Args:
            n: scaling factor
        """
        super().__init__()
        self.name = "SRGAN"
        self.scale = n

        self.transform = v2.Compose([
            v2.PILToTensor(),
            v2.Lambda(lambda x: x / 255.0)
        ])
        
        self.expand = nn.Sequential(
            nn.Conv2d(3, 64, 9, stride=1, padding='same'),
            nn.PReLU()
        )

        self.residual_blocks = nn.Sequential()
        for _ in range(16):
            self.residual_blocks.append(ResBlock())

        self.residual_blocks.append(nn.Conv2d(64, 64, 3, stride=1, padding='same'))
        self.residual_blocks.append(nn.BatchNorm2d(64))

        self.upscaling_head = nn.Sequential()
        for _ in range(int(math.log2(n))):
            self.upscaling_head.append(nn.Conv2d(64, 256, 3, stride=1, padding='same'))
            self.upscaling_head.append(nn.PixelShuffle(2))
            self.upscaling_head.append(nn.PReLU())
            
        self.upscaling_head.append(nn.Conv2d(64, 3, 9, stride=1, padding='same'))
    
    def forward(self, image: str):
        image = Image.open(image).convert("RGB")
        inp = self.transform(image).to(device)
        x = self.expand(inp[None])
        out = (((self.upscaling_head(self.residual_blocks(x) + x)+1.0)/2.0).clamp(0.0, 1.0) * 255.0).squeeze()
        img = Image.fromarray(out.permute(1, 2, 0).to(torch.uint8).cpu().numpy())
        os.makedirs("output", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img.save(f"output/{timestamp}_X{self.scale}.png")





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
        self.name = "RCAN"
        self.scale = n

        self.DIV2K_RGB = torch.tensor([0.44882884613943946, 0.43713809810624193, 0.4040371984052683]).view(1, 3, 1, 1).to(device)

        self.transform = v2.Compose([
            v2.PILToTensor(),
            v2.Lambda(lambda x: x / 255.0)
        ])

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
        image = Image.open(x).convert("RGB")
        inp = self.transform(image).to(device)

        out = ((self.upscaling_head(self.feature_extractor(inp[None]-self.DIV2K_RGB)) + self.DIV2K_RGB).clamp(0.0, 1.0) * 255.0).squeeze()

        img = Image.fromarray(out.permute(1, 2, 0).to(torch.uint8).cpu().numpy())
        os.makedirs("output", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img.save(f"output/{timestamp}_X{self.scale}.png")