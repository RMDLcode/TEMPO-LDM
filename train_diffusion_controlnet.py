"""
ControlNet training on dummy longitudinal latent data.
All dependencies are self-contained; no real data required.
The diffusion model (UNet) is frozen; only ControlNet is trained.

Matches original code:
- controlnet_cond = [starting_z (9) + starting_time (3) + target_time (1)] = 13 channels
- context = date_target (7 features) passed to UNet
"""
import os
import argparse
import warnings
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from monai import transforms
from generative.networks.schedulers import DDPMScheduler

from models import climb

warnings.filterwarnings("ignore")
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class DummyLongitudinalDataset(Dataset):
    """
    Generate synthetic longitudinal latent data.
    - source_latent: 3 timepoints, each (3, 12, 32, 64) -> concatenated to (9, 12, 32, 64)
    - target_latent: 1 timepoint, shape (3, 12, 32, 64)
    - date_source: 3 timepoints × 7 clinical features, shape (3, 7)
    - date_target: 1 timepoint × 7 clinical features, shape (1, 7)
    """
    def __init__(self, num_samples=1000, latent_shape=(3, 12, 32, 64), context_dim=7):
        self.num_samples = num_samples
        self.latent_shape = latent_shape
        self.context_dim = context_dim

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 3 timepoints for source, each 3 channels -> concatenate to 9 channels
        source_timepoints = [torch.randn(*self.latent_shape, dtype=torch.float32) for _ in range(3)]
        source_latent = torch.cat(source_timepoints, dim=0)  # (9, 12, 32, 64)

        # Target latent: 1 timepoint, 3 channels
        target_latent = torch.randn(*self.latent_shape, dtype=torch.float32)  # (3, 12, 32, 64)

        # Clinical features (7-dimensional, values in [0,1])
        date_source = torch.rand(3, self.context_dim, dtype=torch.float32)  # (3,7)
        date_target = torch.rand(1, self.context_dim, dtype=torch.float32)  # (1,7)

        return {
            "source_latent": source_latent,
            "target_latent": target_latent,
            "date_source": date_source,
            "date_target": date_target
        }


def train(args):
    # ---------- Data Loading ----------
    input_num = 3
    pad_transform_3 = transforms.SpatialPad(spatial_size=(input_num * 3, 8, 32, 64), mode="constant")  # 3*3=9
    pad_transform = transforms.SpatialPad(spatial_size=(3, 8, 32, 64), mode="constant")               # 3

    train_set = DummyLongitudinalDataset(num_samples=args.num_train, latent_shape=(3, 6, 32, 64), context_dim=7)
    valid_set = DummyLongitudinalDataset(num_samples=args.num_valid, latent_shape=(3, 6, 32, 64), context_dim=7)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # ---------- Models ----------
    # Load pre-trained diffusion model (UNet) - frozen
    diffusion = climb.init_latent_diffusion_4(args.diff_ckpt, map_location=DEVICE)
    diffusion.to(DEVICE)

    # Initialize ControlNet with input_T=True (to include target time)
    controlnet = climb.init_controlnet_4(input_T=True, multi_heads=True)
    controlnet.load_state_dict(diffusion.state_dict(), strict=False)
    controlnet.to(DEVICE)

    # Freeze diffusion
    for p in diffusion.parameters():
        p.requires_grad = False
    diffusion.eval()

    # ---------- Scheduler & Optimizer ----------
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        schedule='scaled_linear_beta',
        beta_start=0.0015,
        beta_end=0.0205
    )

    optimizer = AdamW(controlnet.parameters(), lr=args.lr)
    scaler = GradScaler()
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.n_epochs)

    # ---------- Training Setup ----------
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(args.output_dir)
    scale_factor = 1.0

    min_loss = float('inf')
    start_epoch = 0
    global_counter = {'train': 0, 'valid': 0}

    # ---------- Main Training Loop ----------
    for epoch in range(start_epoch, args.n_epochs):
        for mode in ['train', 'valid']:
            loader = train_loader if mode == 'train' else valid_loader
            controlnet.train() if mode == 'train' else controlnet.eval()
            diffusion.eval()

            epoch_loss = 0.0
            progress_bar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch} {mode}")

            for step, batch in progress_bar:
                with torch.set_grad_enabled(mode == 'train'):
                    with autocast(enabled=True):
                        # Load data
                        source_latent = batch['source_latent'].to(DEVICE) * scale_factor   # [B, 9, 12, 32, 64]
                        target_latent = batch['target_latent'].to(DEVICE) * scale_factor   # [B, 3, 12, 32, 64]
                        date_source = batch['date_source'].to(DEVICE)                     # [B, 3, 7]
                        date_target = batch['date_target'].to(DEVICE)                     # [B, 1, 7]

                        B = source_latent.shape[0]

                        # Pad spatial dimensions (depth 12 -> 16)
                        source_latent = pad_transform_3(source_latent)  # [B, 9, 16, 32, 64]
                        target_latent = pad_transform(target_latent)    # [B, 3, 16, 32, 64]

                        # ----- Extract first clinical feature (time) from source and target -----
                        # Source: take first feature from each of 3 timepoints -> [B, 3, 1]
                        starting_time = date_source[:, :, 0:1]  # [B, 3, 1]
                        # Target: take first feature -> [B, 1, 1]
                        target_time = date_target[:, :, 0:1]    # [B, 1, 1]

                        # Expand to spatial dimensions
                        # starting_time: [B,3,1] -> [B,3,16,32,64]
                        starting_time_expanded = starting_time.reshape(B, 3, 1, 1, 1).expand(-1, -1, *source_latent.shape[-3:])
                        # target_time: [B,1,1] -> [B,1,16,32,64]
                        target_time_expanded = target_time.reshape(B, 1, 1, 1, 1).expand(-1, -1, *source_latent.shape[-3:])

                        # Build controlnet_cond: source_latent (9) + starting_time (3) + target_time (1) = 13 channels
                        controlnet_cond = torch.cat([
                            source_latent,
                            starting_time_expanded,
                            target_time_expanded
                        ], dim=1)  # [B, 13, 16, 32, 64]

                        # ----- Diffusion forward -----
                        noise = torch.randn_like(target_latent).to(DEVICE)
                        timesteps = torch.randint(0, scheduler.num_train_timesteps, (B,), device=DEVICE).long()
                        noised_target = scheduler.add_noise(target_latent, noise=noise, timesteps=timesteps)

                        # ----- ControlNet forward -----
                        # context = date_target (full 7 features) for cross-attention
                        down_h, mid_h = controlnet(
                            x=noised_target.float(),
                            timesteps=timesteps,
                            context=date_target.float(),          # [B,1,7]
                            controlnet_cond=controlnet_cond.float()
                        )

                        # ----- UNet forward with residuals -----
                        noise_pred, _ = diffusion(
                            x=noised_target.float(),
                            timesteps=timesteps,
                            context=date_target.float(),
                            down_block_additional_residuals=down_h,
                            mid_block_additional_residual=mid_h
                        )

                        # ----- Loss (MSE + L1) -----
                        mse_loss = F.mse_loss(noise_pred.float(), noise.float())
                        l1_loss = F.l1_loss(noise_pred.float(), noise.float())
                        loss = args.mse_weight * mse_loss + args.l1_weight * l1_loss

                    if mode == 'train':
                        scaler.scale(loss).backward()
                        torch.nn.utils.clip_grad_norm_(controlnet.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)

                    epoch_loss += loss.item()
                    progress_bar.set_postfix({"loss": f"{epoch_loss / (step + 1):.4f}"})
                    global_counter[mode] += 1

            avg_loss = epoch_loss / len(loader)
            writer.add_scalar(f'{mode}/epoch_loss', avg_loss, epoch)

            if mode == 'valid':
                print(f"Epoch {epoch} Validation Loss: {avg_loss:.4f}, LR: {lr_scheduler.get_last_lr()[0]:.6f}")
                if avg_loss < min_loss:
                    min_loss = avg_loss
                    torch.save(controlnet.state_dict(), os.path.join(args.output_dir, 'controlnet_best.pth'))
                    torch.save({
                        'epoch': epoch,
                        'controlnet_state': controlnet.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scaler': scaler.state_dict(),
                        'min_loss': min_loss,
                        'lr_scheduler': lr_scheduler.state_dict()
                    }, os.path.join(args.output_dir, 'checkpoint.pth'))

        lr_scheduler.step()
        torch.save(controlnet.state_dict(), os.path.join(args.output_dir, 'controlnet_last.pth'))

    print("Training completed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train ControlNet on dummy longitudinal latent data")
    parser.add_argument('--output_dir', type=str, default='./controlnet_dummy_output',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--diff_ckpt', type=str, default='./dummy_diffusion_output/diffusion_best.pth',
                        help='Path to pretrained diffusion model checkpoint (frozen backbone)')
    parser.add_argument('--n_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num_train', type=int, default=1000, help='Number of training samples')
    parser.add_argument('--num_valid', type=int, default=200, help='Number of validation samples')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--mse_weight', type=float, default=0.5, help='Weight for MSE loss')
    parser.add_argument('--l1_weight', type=float, default=0.5, help='Weight for L1 loss')
    args = parser.parse_args()

    train(args)