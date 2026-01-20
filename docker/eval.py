import os
import sys
import glob
import argparse
import yaml
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from tqdm.auto import tqdm
from torchvision.transforms import v2
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from architectures import RCAN, ESRGAN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)

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

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SISR model on datasets")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config for overrides")
    parser.add_argument("--model", type=str, choices=["rcan", "esrgan"], default="rcan", help="Which architecture to evaluate")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--scale", type=int, default=2, help="Upscale factor")
    parser.add_argument("--flickr-dir", type=str, default=None, help="Directory (or glob) for Flickr2K validation images")
    parser.add_argument("--pokemon-dir", type=str, default=None, help="Directory (or glob) for pokemon validation images")

     # 1) Read only --config from the original CLI
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str)
    cfg_args, _ = config_parser.parse_known_args(sys.argv[1:])

    # 2) Turn YAML into CLI-style args (so argparse enforces types)
    yaml_argv = []
    if cfg_args.config:
        with open(cfg_args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}
        for k, v in cfg.items():
            key = f"--{k.replace('_','-')}"
            yaml_argv.append(key)
            yaml_argv.append(str(v))

    # 3) Parse "YAML first, then real CLI" so CLI overrides YAML
    return parser.parse_args([*yaml_argv, *sys.argv[1:]])

def _gather_paths(path_or_glob: str) -> list[str]:
    if not path_or_glob:
        return []
    
    return sorted(glob.glob(os.path.join(path_or_glob, "*.png")))

def _load_checkpoint_into(model: nn.Module, ckpt_path: str, device):
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=device)
        # possible keys
        for key in ("model_state_dict", "generator_state_dict"):
            if key in ckpt:
                state = ckpt[key]
                model.load_state_dict(state)
       
    return model

def main():
    args = parse_args()

    # Prepare datasets (user can fill directories/globs in YAML or CLI)
    flickr_paths = _gather_paths(args.flickr_dir)
    pokemon_paths = _gather_paths(args.pokemon_dir)

    if not flickr_paths and not pokemon_paths:
        print("No dataset paths provided (use--config YAML). Exiting.")
        return

    # build model
    if args.model == "rcan":
        model_inst = RCAN(args.scale)
    else:
        model_inst = ESRGAN(args.scale)

    model = model_inst.to(device)
    if args.checkpoint:
        model = _load_checkpoint_into(model, args.checkpoint, device)

    model.eval()

    # if flickr_paths:
        # print(f"Evaluating {args.model.upper()} on Flickr2K ({len(flickr_paths)} images), scale={args.scale}")
        # psnr_val, ssim_val, lpips_val = calc_metrics(model, flickr_paths, args.scale)
        # print(f"Flickr2K -> PSNR: {psnr_val:.4f}, SSIM: {ssim_val:.4f}, LPIPS: {lpips_val:.4f}")

    if pokemon_paths:
        print(f"Evaluating {args.model.upper()} on pokemon ({len(pokemon_paths)} images), scale={args.scale}")
        psnr_val, ssim_val, lpips_val = calc_metrics(model, pokemon_paths, args.scale)
        print(f"pokemon -> PSNR: {psnr_val:.4f}, SSIM: {ssim_val:.4f}, LPIPS: {lpips_val:.4f}")

if __name__ == "__main__":
    main()
