import os
import argparse
import warnings
import torch

import torch.nn as nn
from torch.nn import L1Loss
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

from generative.losses import PerceptualLoss, PatchAdversarialLoss
from models.wasserstein_autoencoder import init_wasserstein_autoencoder_inter, init_patch_discriminator
from models.gradacc import GradientAccumulation

import numpy as np

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Window configurations for multi-window loss (CT-specific)
WINDOW_CONFIGS = [
    {'name': 'full', 'low': -1000.0, 'high': 1500.0, 'weight': 0.2},
    {'name': 'soft_tissue', 'low': -160.0, 'high': 240.0, 'weight': 0.5},
    {'name': 'brain', 'low': -10.0, 'high': 70.0, 'weight': 0.3}
]


def rand_projections(embedding_dim, num_samples=50):
    """Generate random projections for Sliced Wasserstein Distance."""
    projections = []
    for _ in range(num_samples):
        w = np.random.normal(size=embedding_dim)
        norm = np.sqrt((w ** 2).sum())
        projections.append(w / norm)
    return torch.tensor(projections, dtype=torch.float32)


def sliced_wasserstein_distance(encoded_samples, distribution_samples,
                                num_projections=50, p=2, device='cpu'):
    """Compute Sliced Wasserstein Distance between encoded and prior samples."""
    embedding_dim = distribution_samples.size(1)
    projections = rand_projections(embedding_dim, num_projections).to(device)
    encoded_proj = encoded_samples.matmul(projections.transpose(0, 1))
    dist_proj = distribution_samples.matmul(projections.transpose(0, 1))
    wasserstein_distance = (torch.sort(encoded_proj, dim=1)[0] -
                            torch.sort(dist_proj, dim=1)[0])
    wasserstein_distance = torch.pow(wasserstein_distance + 1e-8, p)
    return wasserstein_distance.mean()


def compute_window_loss(images, reconstruction, masks=None, mask_weight=0.1,
                        perceptual_weight=0.3, l1_loss_fn=None, perc_loss_fn=None):
    """
    Compute multi-window L1 and perceptual losses.
    Assumes images and reconstruction are in [0,1] (mapped from [-1000,1500]).
    """
    images_real = images * 2500.0 - 1000.0
    recon_real = reconstruction * 2500.0 - 1000.0

    rec_loss_total = 0.0
    per_loss_total = 0.0

    for window in WINDOW_CONFIGS:
        img_clamped = torch.clamp(images_real, window['low'], window['high'])
        recon_clamped = torch.clamp(recon_real, window['low'], window['high'])

        img_win = (img_clamped - window['low']) / (window['high'] - window['low'])
        recon_win = (recon_clamped - window['low']) / (window['high'] - window['low'])

        current_rec_loss = l1_loss_fn(recon_win, img_win)

        if masks is not None:
            mask_rec_loss = l1_loss_fn(recon_win * masks, img_win * masks) * mask_weight
            current_rec_loss += mask_rec_loss

        if perceptual_weight > 0:
            current_per_loss = perc_loss_fn(recon_win, img_win)
            if masks is not None:
                mask_per_loss = perc_loss_fn(recon_win * masks, img_win * masks) * mask_weight
                current_per_loss += mask_per_loss
        else:
            current_per_loss = 0.0

        rec_loss_total += current_rec_loss * window['weight']
        per_loss_total += current_per_loss * window['weight'] * perceptual_weight

    return rec_loss_total, per_loss_total


class DummyDataset(Dataset):
    """Generate random 3D volumes for demonstration. Adjust shape to match your model."""
    def __init__(self, num_samples=100, shape=(1, 96, 256, 512), use_mask=False):
        self.num_samples = num_samples
        self.shape = shape
        self.use_mask = use_mask

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = torch.randn(self.shape, dtype=torch.float32) * 0.5 + 0.5
        img = torch.clamp(img, 0, 1)
        sample = {"img": img}
        if self.use_mask:
            mask = torch.randint(0, 2, (1, *self.shape[1:]), dtype=torch.float32)
            sample["mask"] = mask
        return sample


def train(args, train_loader, valid_loader):
    """Main training loop."""
    # Initialize models
    autoencoder = init_wasserstein_autoencoder_inter(args.swae_ckpt).to(DEVICE)
    discriminator = init_patch_discriminator(args.disc_ckpt).to(DEVICE)

    # Loss functions
    l1_loss_fn = L1Loss()
    adv_loss_fn = PatchAdversarialLoss(criterion="least_squares")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        perc_loss_fn = PerceptualLoss(spatial_dims=3,
                                      network_type="squeeze",
                                      is_fake_3d=True,
                                      fake_3d_ratio=0.2).to(DEVICE)

    # Optimizers and schedulers
    optimizer_g = torch.optim.Adam(autoencoder.parameters(), lr=args.lr)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr)
    scheduler_g = CosineAnnealingLR(optimizer_g, T_max=args.n_epochs, eta_min=1e-6)
    scheduler_d = CosineAnnealingLR(optimizer_d, T_max=args.n_epochs, eta_min=1e-6)

    # Gradient accumulators (using GradScaler for mixed precision)
    gradacc_g = GradientAccumulation(
        actual_batch_size=args.batch_size,
        expect_batch_size=args.max_batch_size,
        loader_len=len(train_loader),
        optimizer=optimizer_g,
        grad_scaler=GradScaler(),
        clip_grad=2.0,
        clip_mode="norm"
    )
    gradacc_d = GradientAccumulation(
        actual_batch_size=args.batch_size,
        expect_batch_size=args.max_batch_size,
        loader_len=len(train_loader),
        optimizer=optimizer_d,
        grad_scaler=GradScaler(),
        clip_grad=1.0,
        clip_mode="norm"
    )

    # Prior samples for SWD (latent dimension depends on your model)
    # Here we assume fixed latent_dim = 3*12*32*64 as in original code
    D, H, W = args.input_shape
    assert D % 8 == 0 and H % 8 == 0 and W % 8 == 0, \
        "Input spatial dimensions must be divisible by 8 (due to 3 downsampling layers)."
    latent_dim = 3 * (D // 8) * (H // 8) * (W // 8)
    # latent_dim = 3 * 12 * 32 * 64
    prior_samples = torch.randn(1, latent_dim).to(DEVICE)

    os.makedirs(args.output_dir, exist_ok=True)
    min_loss = float('inf')
    iteration = 0
    start_epoch = 0
    use_mask_loss = args.use_mask
    initial_mask_weight = 0.1
    valid_mask_weight = 0.1
    mask_warmup_epochs = 10

    # Resume from checkpoint if exists
    if args.resume and os.path.exists(os.path.join(args.output_dir, 'checkpoint_last.pth')):
        checkpoint = torch.load(os.path.join(args.output_dir, 'checkpoint_last.pth'))
        autoencoder.load_state_dict(checkpoint['autoencoder_state'])
        discriminator.load_state_dict(checkpoint['discriminator_state'])
        optimizer_g.load_state_dict(checkpoint['optimizer_g_state'])
        optimizer_d.load_state_dict(checkpoint['optimizer_d_state'])
        scheduler_g.load_state_dict(checkpoint['scheduler_g_state'])
        scheduler_d.load_state_dict(checkpoint['scheduler_d_state'])
        start_epoch = checkpoint['epoch'] + 1
        min_loss = checkpoint['min_loss']
        iteration = checkpoint['iteration']
        print(f"Resuming training from epoch {start_epoch}")

    # Progressive batch strategy (optional, kept from original)
    max_iterations_init = 2000 // args.batch_size
    max_iterations_upper = len(train_loader)
    increment_interval = 10
    increment_step = 100 // args.batch_size

    for epoch in range(start_epoch, args.n_epochs):
        print(f'Starting epoch {epoch}')
        mask_loss_weight = min(initial_mask_weight * (epoch / mask_warmup_epochs), 1.0) if use_mask_loss else 0.0

        autoencoder.train()
        discriminator.train()

        current_max_iter = min(
            max_iterations_init + (epoch // increment_interval) * increment_step,
            max_iterations_upper
        )
        print(f'Epoch {epoch}: using {current_max_iter} batches')

        for step, batch in enumerate(train_loader):
            if step >= current_max_iter:
                break
            iteration += 1

            # --- Train Generator ---
            with autocast(enabled=True):
                images = batch["img"].to(DEVICE)
                masks = batch["mask"].to(DEVICE) if use_mask_loss else None

                z = autoencoder.encode(images)
                reconstruction = autoencoder.decode(z)

                rec_loss, per_loss = compute_window_loss(
                    images, reconstruction,
                    masks=masks,
                    mask_weight=mask_loss_weight,
                    perceptual_weight=args.perceptual_weight,
                    l1_loss_fn=l1_loss_fn,
                    perc_loss_fn=perc_loss_fn
                )

                logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                gen_loss = args.adv_weight * adv_loss_fn(logits_fake, target_is_real=True, for_discriminator=False)

                swd_loss = args.lambda_weight * sliced_wasserstein_distance(
                    z.view(images.size(0), -1),
                    prior_samples.repeat(images.size(0), 1),
                    args.num_projections,
                    device=DEVICE
                )

                loss_g = rec_loss + per_loss + gen_loss + swd_loss

            gradacc_g.step(loss_g, step)

            # --- Train Discriminator ---
            with autocast(enabled=True):
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                d_loss_fake = adv_loss_fn(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                d_loss_real = adv_loss_fn(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (d_loss_fake + d_loss_real) * 0.5
                loss_d = args.adv_weight * discriminator_loss

            gradacc_d.step(loss_d, step)

            if step % 10 == 0:
                print(f'Step {step}: G_loss={loss_g.item():.4f}, D_loss={loss_d.item():.4f}')

        # End of epoch: update schedulers and validate
        scheduler_g.step()
        scheduler_d.step()

        autoencoder.eval()
        discriminator.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                images = batch["img"].to(DEVICE)
                masks = batch["mask"].to(DEVICE) if use_mask_loss else None
                z = autoencoder.encode(images)
                reconstruction = autoencoder.decode(z)
                rec_loss, per_loss = compute_window_loss(
                    images, reconstruction,
                    masks=masks,
                    mask_weight=valid_mask_weight,
                    perceptual_weight=args.perceptual_weight,
                    l1_loss_fn=l1_loss_fn,
                    perc_loss_fn=perc_loss_fn
                )
                val_loss_total += (rec_loss + per_loss).item()

        avg_val_loss = val_loss_total / len(valid_loader)
        print(f'Epoch {epoch} Validation loss: {avg_val_loss:.4f}, LR: {optimizer_g.param_groups[0]["lr"]:.6f}')

        # Save best and last checkpoints
        if avg_val_loss < min_loss:
            min_loss = avg_val_loss
            torch.save(discriminator.state_dict(), os.path.join(args.output_dir, 'discriminator_best.pth'))
            torch.save(autoencoder.state_dict(), os.path.join(args.output_dir, 'swae_best.pth'))
            torch.save({
                'epoch': epoch,
                'autoencoder_state': autoencoder.state_dict(),
                'discriminator_state': discriminator.state_dict(),
                'optimizer_g_state': optimizer_g.state_dict(),
                'optimizer_d_state': optimizer_d.state_dict(),
                'scheduler_g_state': scheduler_g.state_dict(),
                'scheduler_d_state': scheduler_d.state_dict(),
                'min_loss': min_loss,
                'iteration': iteration
            }, os.path.join(args.output_dir, 'checkpoint.pth'))

        torch.save({
            'epoch': epoch,
            'autoencoder_state': autoencoder.state_dict(),
            'discriminator_state': discriminator.state_dict(),
            'optimizer_g_state': optimizer_g.state_dict(),
            'optimizer_d_state': optimizer_d.state_dict(),
            'scheduler_g_state': scheduler_g.state_dict(),
            'scheduler_d_state': scheduler_d.state_dict(),
            'min_loss': min_loss,
            'iteration': iteration
        }, os.path.join(args.output_dir, 'checkpoint_last.pth'))

        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='./output_dummy', help='Output directory')
    parser.add_argument('--swae_ckpt', default=None, type=str, help='Pretrained WAE checkpoint')
    parser.add_argument('--disc_ckpt', default=None, type=str, help='Pretrained discriminator checkpoint')
    parser.add_argument('--n_epochs', default=100, type=int, help='Number of epochs')
    parser.add_argument('--batch_size', default=1, type=int, help='Mini-batch size')
    parser.add_argument('--max_batch_size', default=2, type=int, help='Effective batch size (gradient accumulation target)')
    parser.add_argument('--num_projections', default=100, type=int, help='Number of projections for SWD')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--adv_weight', default=0.3, type=float, help='Adversarial loss weight')
    parser.add_argument('--perceptual_weight', default=0.3, type=float, help='Perceptual loss weight')
    parser.add_argument('--lambda_weight', default=0.1, type=float, help='SWD loss weight')
    parser.add_argument('--use_mask', action='store_true', help='Use mask loss (dummy masks generated)')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--input_shape', nargs=3, type=int, default=[48, 256, 512],
                        help='Input volume shape (D, H, W) - must match your model\'s expected input')
    args = parser.parse_args()

    # Create dummy datasets with user-specified shape (default matches typical model)
    shape = (1, args.input_shape[0], args.input_shape[1], args.input_shape[2])
    train_set = DummyDataset(num_samples=10, shape=shape, use_mask=args.use_mask)
    valid_set = DummyDataset(num_samples=2, shape=shape, use_mask=args.use_mask)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f'Using dummy data with shape {shape}')
    print(f'Device: {DEVICE}')

    train(args, train_loader, valid_loader)
