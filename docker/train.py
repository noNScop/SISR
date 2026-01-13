import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from datasets import RCAN_Dataset, HR_valid_paths, HR_train_paths
from training_engine import train
from architectures import RCAN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

valid_ds = RCAN_Dataset(HR_valid_paths, 2, ram_limit_gb=8)
train_ds = RCAN_Dataset(HR_train_paths, 2, ram_limit_gb=32)

train_dl = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=os.cpu_count()-1)
valid_dl = DataLoader(valid_ds, batch_size=16, shuffle=False, num_workers=os.cpu_count()-1)

model = torch.compile(RCAN(2).to(device))
loss_fn = nn.L1Loss()
optimizer = Adam(model.parameters(), lr=1e-4)
scheduler = StepLR(optimizer, step_size=400, gamma=0.5)

train(model, train_dl, valid_dl, optimizer, scheduler, loss_fn, 2000)