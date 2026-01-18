import os
import csv
import torch
from pathlib import Path
from tqdm.auto import tqdm
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import StepLR
from eval_tools import psnr, ssim, lpips

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_step(model, dataloader, optimizer, loss_fn, scaler):
    avg_psnr = 0
    avg_ssim = 0
    model.train()

    for batch, target in dataloader:
        optimizer.zero_grad(set_to_none=True)
        
        batch, target = batch.to(device), target.to(device)

        with autocast('cuda'):
            logits = model(batch)
            loss = loss_fn(logits, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        logits = logits.clamp(0.0, 1.0)
        target = target.clamp(0.0, 1.0)
        
        avg_psnr += psnr(logits, target).item()
        avg_ssim += ssim(logits, target).item()
        
    avg_psnr /= len(dataloader)
    avg_ssim /= len(dataloader)
    return avg_psnr, avg_ssim

def valid_step(model, dataloader, loss_fn):
    avg_psnr = 0
    avg_ssim = 0
    avg_lpips = 0
    model.eval()

    with torch.inference_mode():
        for batch, target in dataloader:
            batch, target = batch.to(device), target.to(device)

            logits = model(batch)
            
            logits = logits.clamp(0.0, 1.0)
            target = target.clamp(0.0, 1.0)
            
            avg_psnr += psnr(logits, target).item()
            avg_ssim += ssim(logits, target).item()
            avg_lpips += lpips(logits, target).item()

    avg_psnr /= len(dataloader)
    avg_ssim /= len(dataloader)
    avg_lpips /= len(dataloader)

        
    return avg_psnr, avg_ssim, avg_lpips

def pretrain_esrgan(model, train_dl, valid_dl, optimizer, scheduler: StepLR, loss_fn, epochs, start_checkpoint=None):
    os.makedirs('../tmp_model_checkpoints', exist_ok=True)
    counter = 0 # count epochs without printing training stats
    scaler = GradScaler('cuda')
    
    if start_checkpoint:
        start_epoch = start_checkpoint['epoch']
        best_psnr = start_checkpoint['best_psnr']
        best_ssim = start_checkpoint['best_ssim']
        best_lpips = start_checkpoint['best_lpips']
        scaler.load_state_dict(start_checkpoint['scaler_state_dict'])
    else:
        start_epoch = 0
        best_psnr = 0
        best_ssim = 0
        best_lpips = float('inf')
        
    log_freq = (epochs - start_epoch) // 20 # how often to print stats when no progress is made
    
    for epoch in tqdm(range(start_epoch, epochs), desc="Epochs"):
        counter += 1
        train_psnr, train_ssim = train_step(
            model,
            train_dl,
            optimizer,
            loss_fn,
            scaler
        )

        valid_psnr, valid_ssim, valid_lpips = valid_step(
            model,
            valid_dl,
            loss_fn,
        )

        scheduler.step()

        progress = False
        
        if valid_psnr > best_psnr:
            progress = True
            best_psnr = valid_psnr
            checkpoint = {
                'epoch': epoch,
                'best_psnr': best_psnr,
                'best_ssim': best_ssim,
                'best_lpips': best_lpips,
                'model_state_dict': model._orig_mod.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict()
            }
            torch.save(checkpoint, f'../tmp_model_checkpoints/best_psnr.pth')

        if valid_ssim > best_ssim:
            progress = True
            best_ssim = valid_ssim
            checkpoint = {
                'epoch': epoch,
                'best_psnr': best_psnr,
                'best_ssim': best_ssim,
                'best_lpips': best_lpips,
                'model_state_dict': model._orig_mod.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict()
            }
            torch.save(checkpoint, f'../tmp_model_checkpoints/best_ssim.pth')

        if valid_lpips < best_lpips:
            progress = True
            best_lpips = valid_lpips
            checkpoint = {
                'epoch': epoch,
                'best_psnr': best_psnr,
                'best_ssim': best_ssim,
                'best_lpips': best_lpips,
                'model_state_dict': model._orig_mod.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict()
            }
            torch.save(checkpoint, f'../tmp_model_checkpoints/best_lpips.pth')

        if epoch == epochs-1:
            checkpoint = {
                'epoch': epoch,
                'best_psnr': best_psnr,
                'best_ssim': best_ssim,
                'best_lpips': best_lpips,
                'model_state_dict': model._orig_mod.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict()
            }
            torch.save(checkpoint, f'../tmp_model_checkpoints/last.pth')
            
        if progress or counter >= log_freq:
            counter = 0
            print(
                f"Epoch: {epoch+1} | "
                f"learning rate: {scheduler.get_last_lr()[0]:.6f} | "
                f"[train] PSNR: {train_psnr:.4f} | "
                f"[train] SSIM: {train_ssim:.4f} | "
                f"[valid] PSNR: {valid_psnr:.4f} | "
                f"[valid] SSIM: {valid_ssim:.4f} | "
                f"[valid] LPIPS: {valid_lpips:.4f}"
            )