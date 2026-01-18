import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from datasets import SRDataset, HR_valid_paths, HR_train_paths
from rcan_training import train_rcan
from esrgan_pretraining import pretrain_esrgan
from esrgan_training import train_esrgan
from architectures import RCAN, ESRGAN, RaD

import argparse
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description="Train SISR model")
    # data/model
    parser.add_argument("--scale", type=int, choices=[2], default=2, help="Upscale factor")
    parser.add_argument("--ram-limit-train", type=int, default=32, help="RAM limit (GB) for training dataset cache")
    parser.add_argument("--ram-limit-valid", type=int, default=8, help="RAM limit (GB) for validation dataset cache")
    parser.add_argument("--model", type=str, default="rcan", choices=["rcan","esrgan_pretrain","esrgan"], help="Model to train")
    # training
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--step-size", type=int, default=400)
    parser.add_argument("--pretrained-path", type=str, default="./tmp_model_checkpoints/ESRGAN_pretrained_X2.pth")
    # config file
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config for overrides")

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

def main():
    args = parse_args()
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    valid_ds = SRDataset(HR_valid_paths, args.scale, ram_limit_gb=args.ram_limit_valid)
    train_ds = SRDataset(HR_train_paths, args.scale, ram_limit_gb=args.ram_limit_train)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=os.cpu_count()-1)
    valid_dl = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=os.cpu_count()-1)

    if args.model == "rcan":
        model = torch.compile(RCAN(2).to(device))
        loss_fn = nn.L1Loss()
        optimizer = Adam(model.parameters(), lr=1e-4)
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=0.5)

        train_rcan(model, train_dl, valid_dl, optimizer, scheduler, loss_fn, args.epochs)
    elif args.model == "esrgan_pretrain":
        model = torch.compile(ESRGAN(2).to(device))
        loss_fn = nn.L1Loss()
        optimizer = Adam(model.parameters(), 2e-4)
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=0.5)

        pretrain_esrgan(model, train_dl, valid_dl, optimizer, scheduler, loss_fn, args.epochs)
    else:
        l1_loss = nn.L1Loss()
        pretrained_path = Path(args.pretrained_path)
        if not pretrained_path.exists():
            raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_path}")

        try:
            checkpoint = torch.load(pretrained_path, map_location=device)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint {pretrained_path}: {e}") from e

        generator = ESRGAN(2).to(device)
        try:
            # support checkpoints that store state dict under different keys
            state = checkpoint.get('model_state_dict', checkpoint)
            generator.load_state_dict(state)
        except Exception as e:
            raise RuntimeError(f"Failed to load model state_dict from {pretrained_path}: {e}") from e

        discriminator = RaD().to(device)

        generator_opt = Adam(generator.parameters(), lr=1e-4)
        discriminator_opt = Adam(discriminator.parameters(), lr=1e-4)

        generator_scheduler = StepLR(generator_opt, step_size=1, gamma=0.5)
        discriminator_scheduler = StepLR(discriminator_opt, step_size=1, gamma=0.5)

        train_esrgan(generator, discriminator, train_dl, valid_dl, generator_opt, discriminator_opt, generator_scheduler, discriminator_scheduler, l1_loss, args.epochs)

if __name__ == "__main__":
    main()