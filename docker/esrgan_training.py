import os
import csv
import torch
import torchvision
from pathlib import Path
from tqdm.auto import tqdm
from torchvision.models import vgg19
from eval_tools import psnr, ssim, lpips
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import StepLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vgg = vgg19(weights=torchvision.models.VGG19_Weights.DEFAULT).to(device)
vgg54 = vgg.features[:35]
vgg54.eval()
vgg_transform = torchvision.models.VGG19_Weights.IMAGENET1K_V1.transforms()

def train_generator(batch, xr, generator, discriminator, optimizer, scaler, l1_loss):
    eta = 0.01
    lmbda = 0.005
    generator.train()
    discriminator.eval()
    
    optimizer.zero_grad(set_to_none=True)

    with autocast('cuda'):
        xf = generator(batch)

        d_xf_xr = discriminator(xf, xr)
        d_xr_xf = discriminator(xr, xf)

        eps = 1e-2
        d_xf_xr = d_xf_xr.clamp(eps, 1 - eps)
        d_xr_xf = d_xr_xf.clamp(eps, 1 - eps)
        
        RaD_loss = - torch.log(1 - d_xr_xf).mean() - torch.log(d_xf_xr).mean()
        
        loss = torch.mean((vgg54(vgg_transform(xr)) - vgg54(vgg_transform(xf)))**2) + lmbda * RaD_loss + eta * l1_loss(xf, xr)

        if loss == float('inf'):
            print(f"RAD: {RaD_loss} | L1:  {l1_loss(xf, xr)} | VGG: {vgg54(vgg_transform(xr)) - vgg54(vgg_transform(xf))}")

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    return loss.item()

def train_discriminator(LR, xr, generator, discriminator, optimizer):
    generator.eval()
    discriminator.train()
    optimizer.zero_grad(set_to_none=True)
    
    with torch.no_grad():
        xf = generator(LR).detach()

    d_xf_xr = discriminator(xf, xr)
    d_xr_xf = discriminator(xr, xf)

    eps = 1e-4
    d_xf_xr = d_xf_xr.clamp(eps, 1 - eps)
    d_xr_xf = d_xr_xf.clamp(eps, 1 - eps)

    loss = - torch.log(d_xr_xf).mean() - torch.log(1 - d_xf_xr).mean()

    loss.backward()
    optimizer.step()
    
    return loss.item()

def train_step(generator, discriminator, dataloader, generator_opt, discriminator_opt, l1_loss, generator_scaler):
    discriminator_loss = 0
    generator_loss = 0
    
    for batch, target in dataloader:
        batch, target = batch.to(device), target.to(device)
        discriminator_loss += train_discriminator(batch, target, generator, discriminator, discriminator_opt)
        generator_loss += train_generator(batch, target, generator, discriminator, generator_opt, generator_scaler, l1_loss)
        
    discriminator_loss /= len(dataloader)
    generator_loss /= len(dataloader)
    return discriminator_loss, generator_loss

def valid_step(generator, discriminator, dataloader, l1_loss):
    eta = 0.01
    lmbda = 0.005
    generator.eval()
    discriminator.eval()
    discriminator_loss = 0
    generator_loss = 0
    lpips_acc = 0

    with torch.inference_mode():
        for batch, target in dataloader:
            batch, xr = batch.to(device), target.to(device)
            
            xf = generator(batch)
            lpips_acc += lpips(((xf+1.0)/2.0).clamp(0.0, 1.0), ((xr+1.0)/2.0).clamp(0.0, 1.0))

            RaD_loss = - torch.log(1 - discriminator(xr, xf) + 1e-12).mean() - torch.log(discriminator(xf, xr) + 1e-12).mean()
            generator_loss += (torch.mean((vgg54(vgg_transform(xr)) - vgg54(vgg_transform(xf)))**2) + lmbda * RaD_loss + eta * l1_loss(xf, xr)).item()

            discriminator_loss += (- torch.log(discriminator(xr, xf) + 1e-12).mean() - torch.log(1 - discriminator(xf, xr) + 1e-12).mean()).item()
        
        discriminator_loss /= len(dataloader) 
        generator_loss /= len(dataloader)
        lpips_acc /= len(dataloader)


    return discriminator_loss, generator_loss, lpips_acc

def train_esrgan(generator, discriminator, train_dl, valid_dl, generator_opt, discriminator_opt, generator_scheduler: StepLR, 
          discriminator_scheduler: StepLR, l1_loss, epochs, start_checkpoint=None):
    os.makedirs('../tmp_model_checkpoints', exist_ok=True)
    counter = 0 # count epochs without printing training stats
    best_lpips= float('inf')
    generator_scaler = GradScaler('cuda')
    
    if start_checkpoint:
        start_epoch = start_checkpoint['epoch'] + 1
        best_lpips = start_checkpoint['lpips']
        generator_scaler.load_state_dict(start_checkpoint['generator_scaler_state_dict'])
    else:
        start_epoch = 0
        
    log_freq = (epochs - start_epoch) // 100 # how often to print stats when no progress is made
        
    for epoch in tqdm(range(start_epoch, epochs), desc="Epochs"):
        counter += 1
        train_d_loss, train_g_loss = train_step(
            generator,
            discriminator,
            train_dl,
            generator_opt,
            discriminator_opt,
            l1_loss,
            generator_scaler
        )

        valid_d_loss, valid_g_loss, valid_lpips = valid_step(
            generator,
            discriminator,
            valid_dl,
            l1_loss
        )
        if epoch+1 in [int(0.1*epochs), int(0.2*epochs), int(0.4*epochs), int(0.6*epochs)]:
            discriminator_scheduler.step()
            generator_scheduler.step()

        progress = False
        
        if valid_lpips < best_lpips:
            progress = True
            best_lpips = valid_lpips
            checkpoint = {
                'epoch': epoch,
                'lpips': best_lpips,
                'discriminator_state_dict': discriminator.discriminator.state_dict(),
                'generator_state_dict': generator.state_dict(),
                'discriminator_optimizer_state_dict': discriminator_opt.state_dict(),
                'generator_optimizer_state_dict': generator_opt.state_dict(),
                'discriminator_scheduler_state_dict': discriminator_scheduler.state_dict(),
                'generator_scheduler_state_dict': generator_scheduler.state_dict(),
                'generator_scaler_state_dict': generator_scaler.state_dict()
            }
            torch.save(checkpoint, f'../tmp_model_checkpoints/best.pth')

        if epoch == epochs-1:
            checkpoint = {
                'epoch': epoch,
                'lpips': best_lpips,
                'discriminator_state_dict': discriminator.discriminator.state_dict(),
                'generator_state_dict': generator.state_dict(),
                'discriminator_optimizer_state_dict': discriminator_opt.state_dict(),
                'generator_optimizer_state_dict': generator_opt.state_dict(),
                'discriminator_scheduler_state_dict': discriminator_scheduler.state_dict(),
                'generator_scheduler_state_dict': generator_scheduler.state_dict(),
                'generator_scaler_state_dict': generator_scaler.state_dict()
            }
            torch.save(checkpoint, f'../tmp_model_checkpoints/last.pth')
            
        if True or progress or counter >= log_freq:
            counter = 0
            print(
                f"Epoch: {epoch+1} | "
                f"learning rate: {generator_scheduler.get_last_lr()[0]:.6f} | "
                f"[train] generator: {train_g_loss:.4f} | "
                f"[train] discriminator: {train_d_loss:.4f} | "
                f"[valid] generator: {valid_g_loss:.4f} | "
                f"[valid] discriminator: {valid_d_loss:.4f} | "
                f"[valid] lpips: {valid_lpips:.4f}"
            )