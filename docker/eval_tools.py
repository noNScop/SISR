import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from tqdm.auto import tqdm
from torchvision.transforms import v2
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)

transform = v2.Compose([
    v2.PILToTensor(),
    v2.Lambda(lambda x: x / 255.0)
])

def calc_metrics(model: nn.Module, target_ds: list[str], scale: int):
    transform_target = v2.Compose([
        v2.PILToTensor(),
        v2.Lambda(lambda x: x/255.0)
    ])

    transform_input = v2.Compose([
        v2.PILToTensor(),
        v2.Lambda(lambda x: (x / 255.0))
    ])

    psnr_acc = 0
    ssim_acc = 0
    lpips_acc = 0
    failed_lpips = 0

    for i in tqdm(range(len(target_ds)), leave=False):
        target_image = Image.open(target_ds[i]).convert("RGB")
        w, h = target_image.size

        w -= w % scale
        h -= h % scale
        target_image = target_image.crop((0, 0, w, h))
        
        lowres = target_image.resize((w // scale, h // scale), resample=Image.BICUBIC)
        input_tensor = transform_input(lowres).unsqueeze(0).to(device)
        target_tensor = transform_target(target_image).unsqueeze(0).to(device)

        with torch.inference_mode():
            sr = model(input_tensor).clamp(0, 1)

        psnr_acc += psnr(sr, target_tensor).item()
        ssim_acc += ssim(sr, target_tensor).item()
        
        # There are 2 images that cause lpips to fail
        try:
            x = lpips(sr, target_tensor).cpu().item()
            if np.isnan(x):
                failed_lpips += 1
                continue
                
            lpips_acc += x
        except:
            failed_lpips += 1

    lpips_acc /= len(target_ds) - failed_lpips
    psnr_acc /= len(target_ds)
    ssim_acc /= len(target_ds)
    return psnr_acc, ssim_acc, lpips_acc