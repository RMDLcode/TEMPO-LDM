"""
Diffusion model training on dummy latent data with 7 clinical features.
All dependencies are self-contained; no real data required.
"""
import os
import argparse
import warnings
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from monai import transforms
from monai.utils import set_determinism
from generative.networks.schedulers import DDPMScheduler
from generative.inferers import DiffusionInferer

from models import climb  # Your diffusion model initializer

set_determinism(0)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class DummyLatentDataset(Dataset):
    """
    Generate synthetic latent vectors and clinical conditions.
    - latent: random tensor of shape (3, 12, 32, 64)
    - context: 7 random features normalized to [0,1]
    """
    def __init__(self, num_samples=100, latent_shape=(3, 12, 32, 64), context_dim=7):
        self.num_samples = num_samples
        self.latent_shape = latent_shape
        self.context_dim = context_dim

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        latent = torch.randn(*self.latent_shape, dtype=torch.float32)
        context = torch.rand(self.context_dim, dtype=torch.float32)
        return {"latent": latent, "context": context}


def train(args):
    # ---------- Data Loading ----------
    pad_transform = transforms.SpatialPad(spatial_size=(3, 8, 32, 64), mode="constant")

    train_set = DummyLatentDataset(num_samples=200, latent_shape=(3, 6, 32, 64), context_dim=7)
    valid_set = DummyLatentDataset(num_samples=50, latent_shape=(3, 6, 32, 64), context_dim=7)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)

    # ---------- Model ----------
    diffusion = climb.init_latent_diffusion_4(args.diff_ckpt).to(DEVICE)

    # ---------- Scheduler & Inferer ----------
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        schedule='scaled_linear_beta',
        beta_start=0.0015,
        beta_end=0.0205
    )
    inferer = DiffusionInferer(scheduler=scheduler)

    # ---------- Optimizer & LR ----------
    optimizer = AdamW(diffusion.parameters(), lr=args.lr)
    scaler = GradScaler()
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.n_epochs)

    # ---------- Training Setup ----------
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(args.output_dir)
    scale_factor = 1.0  # In practice, compute from data

    min_loss = float('inf')
    start_epoch = 0
    global_counter = {'train': 0, 'valid': 0}

    # ---------- Main Loop ----------
    for epoch in range(start_epoch, args.n_epochs):
        for mode in ['train', 'valid']:
            loader = train_loader if mode == 'train' else valid_loader
            diffusion.train() if mode == 'train' else diffusion.eval()
            epoch_loss = 0.0
            progress_bar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch} {mode}")

            for step, batch in progress_bar:
                with autocast(enabled=True):
                    latents = batch['latent'].to(DEVICE) * scale_factor
                    context = batch['context'].to(DEVICE)
                    if context.dim() == 2:
                        context = context.unsqueeze(1)  # (B, 1, 7)
                    n = latents.shape[0]

                    latents = pad_transform(latents)

                    noise = torch.randn_like(latents).to(DEVICE)
                    timesteps = torch.randint(0, scheduler.num_train_timesteps, (n,), device=DEVICE).long()

                    with torch.set_grad_enabled(mode == 'train'):
                        noise_pred, _ = inferer(
                            inputs=latents,
                            diffusion_model=diffusion,
                            noise=noise,
                            timesteps=timesteps,
                            condition=context,
                            mode='crossattn'
                        )
                        loss = F.mse_loss(noise.float(), noise_pred.float())

                if mode == 'train':
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                epoch_loss += loss.item()
                progress_bar.set_postfix({"loss": f"{epoch_loss / (step + 1):.4f}"})
                global_counter[mode] += 1

            avg_loss = epoch_loss / len(loader)
            if mode == 'valid':
                print(f"Epoch {epoch} Validation Loss: {avg_loss:.4f}, LR: {lr_scheduler.get_last_lr()[0]:.6f}")
                if avg_loss < min_loss:
                    min_loss = avg_loss
                    torch.save(diffusion.state_dict(), os.path.join(args.output_dir, 'diffusion_best.pth'))
                    torch.save({
                        'epoch': epoch,
                        'state_dict': diffusion.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scaler': scaler.state_dict(),
                        'min_loss': min_loss,
                        'lr_scheduler': lr_scheduler.state_dict()
                    }, os.path.join(args.output_dir, 'checkpoint.pth'))
            writer.add_scalar(f'{mode}/epoch_loss', avg_loss, epoch)

        lr_scheduler.step()
        torch.save(diffusion.state_dict(), os.path.join(args.output_dir, 'diffusion_last.pth'))

    print("Training completed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train diffusion model on dummy latent data")
    parser.add_argument('--output_dir', type=str, default='./dummy_diffusion_output',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--diff_ckpt', type=str, default=None,
                        help='Pretrained diffusion model checkpoint (optional)')
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    args = parser.parse_args()

    train(args)