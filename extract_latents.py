import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
from models.wasserstein_autoencoder import init_wasserstein_autoencoder_inter

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract latent features using dummy data")
    parser.add_argument('--output_dir', type=str, default='./dummy_latent_features',
                        help='Directory to save extracted latent .npz files')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of dummy samples to generate')
    parser.add_argument('--input_shape', nargs=3, type=int, default=[48, 256, 512],
                        help='Input volume shape (D, H, W) – must match your model')
    parser.add_argument('--aekl_ckpt', type=str, default='./output_dummy/swae_best.pth',
                        help='Path to pretrained autoencoder checkpoint (optional)')
    args = parser.parse_args()

    # 1. Initialize the model (consistent with training code)
    autoencoder = init_wasserstein_autoencoder_inter(args.aekl_ckpt).to(DEVICE)
    autoencoder.eval()

    # 2. Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # 3. Generate dummy data and extract latent features
    with torch.no_grad():
        for idx in tqdm(range(args.num_samples), desc="Extracting features"):
            # Generate a random volume in [0,1] (same normalization as training)
            dummy_img = torch.rand(1, 1, *args.input_shape, dtype=torch.float32).to(DEVICE)

            # Encode to latent space [1, latent_dim]
            latent = autoencoder.encode(dummy_img)
            latent_np = latent.detach().cpu().numpy()[0]  # shape: (latent_dim,)

            # Save as compressed .npz file (named by sample index)
            save_path = os.path.join(args.output_dir, f"sample_{idx:04d}_latent.npz")
            np.savez_compressed(save_path, data=latent_np)

    print(f"Done. {args.num_samples} latent vectors saved to {args.output_dir}")