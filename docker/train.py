import os
import sys
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from datasets import RCAN_Dataset, HR_valid_paths, HR_train_paths
from training_engine import train
from architectures import RCAN

import argparse
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description="Train SISR model")
    # data/model
    parser.add_argument("--scale", type=int, choices=[2], default=2, help="Upscale factor")
    parser.add_argument("--ram-limit-train", type=int, default=32, help="RAM limit (GB) for training dataset cache")
    parser.add_argument("--ram-limit-valid", type=int, default=8, help="RAM limit (GB) for validation dataset cache")
    # training
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--step-size", type=int, default=400)
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

    valid_ds = RCAN_Dataset(HR_valid_paths, args.scale, ram_limit_gb=args.ram_limit_valid)
    train_ds = RCAN_Dataset(HR_train_paths, args.scale, ram_limit_gb=args.ram_limit_train)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=os.cpu_count()-1)
    valid_dl = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=os.cpu_count()-1)

    model = torch.compile(RCAN(2).to(device))
    loss_fn = nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=0.5)

    train(model, train_dl, valid_dl, optimizer, scheduler, loss_fn, args.epochs)

if __name__ == "__main__":
    main()