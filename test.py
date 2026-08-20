import os
import gc
import numpy as np
import torch
import SimpleITK as sitk
from torch import nn
from tqdm import tqdm
from torch.cuda.amp import autocast
from monai import transforms
from models import wasserstein_autoencoder, climb

# ---------- Import schedulers for sampling ----------
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler
try:
    from diffusers import DPMSolverMultistepScheduler, UniPCMultistepScheduler
except ImportError:
    DPMSolverMultistepScheduler = None
    UniPCMultistepScheduler = None

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---------- Global padding transforms (used in sampling) ----------
input_num = 3
pad_transform_3 = transforms.SpatialPad(spatial_size=(input_num * 3, 8, 32, 64), mode="constant")
pad_transform = transforms.SpatialPad(spatial_size=(3, 8, 32, 64), mode="constant")

# ---------- Helper functions ----------
def get_largest_connected_component(image):
    img = sitk.GetImageFromArray(image)
    mask = sitk.BinaryThreshold(img, lowerThreshold=0.05, upperThreshold=float('inf'))
    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_image = cc_filter.Execute(mask)
    stats_filter = sitk.LabelShapeStatisticsImageFilter()
    stats_filter.Execute(cc_image)
    if stats_filter.GetNumberOfLabels() > 0:
        largest_label = max([label for label in stats_filter.GetLabels()],
                            key=lambda x: stats_filter.GetNumberOfPixels(x))
        largest_mask = sitk.BinaryThreshold(cc_image, lowerThreshold=largest_label, upperThreshold=largest_label)
        img = sitk.Mask(img, largest_mask)
    img = sitk.Cast(img, sitk.sitkFloat32)
    return sitk.GetArrayFromImage(img)


@torch.no_grad()
def sample_latent_using_controlnet(
    diffusion: nn.Module,
    controlnet: nn.Module,
    starting_z: torch.Tensor,
    starting_a: torch.Tensor,
    context: torch.Tensor,
    device: str,
    scale_factor: float = 1.0,
    average_over_n: int = 10,
    num_training_steps: int = 1000,
    num_inference_steps: int = 50,
    schedule: str = 'scaled_linear_beta',
    scheduler_name: str = 'DDIM',
    beta_start: float = 0.0015,
    beta_end: float = 0.0205,
    verbose: bool = False,
    seed: int = None,
    input_T: bool = True,
) -> torch.Tensor:
    if seed is not None:
        torch.manual_seed(seed)
        if device == 'cuda':
            torch.cuda.manual_seed_all(seed)

    # Scheduler setup
    if scheduler_name == 'DDIM':
        scheduler = DDIMScheduler(num_train_timesteps=num_training_steps,
                                  schedule=schedule,
                                  beta_start=beta_start,
                                  beta_end=beta_end,
                                  clip_sample=False)
    elif scheduler_name == 'DDPM':
        scheduler = DDPMScheduler(num_train_timesteps=num_training_steps,
                                  schedule=schedule,
                                  beta_start=beta_start,
                                  beta_end=beta_end,
                                  clip_sample=False)
    elif scheduler_name == 'DPM-Solver++':
        if DPMSolverMultistepScheduler is None:
            raise ImportError("Please install diffusers: pip install diffusers")
        scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=num_training_steps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule="scaled_linear",
            algorithm_type="dpmsolver++",
            solver_order=3,
            use_karras_sigmas=False
        )
    elif scheduler_name == 'UniPC':
        if UniPCMultistepScheduler is None:
            raise ImportError("Please install diffusers: pip install diffusers")
        scheduler = UniPCMultistepScheduler(
            num_train_timesteps=num_training_steps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule="scaled_linear",
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
    scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    # Prepare inputs
    if len(starting_z.shape) == 4:
        starting_z = starting_z.unsqueeze(0).to(device)
    else:
        starting_z = starting_z.to(device)

    starting_z_padded = pad_transform_3(starting_z)
    starting_a = starting_a[:, :, 0:1]
    n, l, c = starting_a.shape[:3]
    concatenating_age = starting_a.reshape(n, l*c, 1, 1, 1).expand(n, l*c, *starting_z_padded.shape[-3:]).to(device)
    if input_T:
        context_a = context[:, :, 0:1]
        concatenating_context = context_a.reshape(n, c, 1, 1, 1).expand(n, c, *starting_z_padded.shape[-3:])
        controlnet_condition = torch.cat([starting_z_padded, concatenating_age, concatenating_context], dim=1)
    else:
        controlnet_condition = torch.cat([starting_z_padded, concatenating_age], dim=1).to(device)

    if len(context.shape) == 2:
        context = context.unsqueeze(0).to(device)
    else:
        context = context.to(device)

    if average_over_n > 1:
        context = context.repeat(average_over_n, 1, 1)
        controlnet_condition = controlnet_condition.repeat(average_over_n, 1, 1, 1, 1)

    z = torch.randn(average_over_n, 3, *starting_z_padded.shape[-3:]).to(device)

    # Sampling loop
    progress_bar = tqdm(scheduler.timesteps) if verbose else scheduler.timesteps
    for t in progress_bar:
        timestep = torch.tensor([t]).repeat(average_over_n).to(device)
        down_h, mid_h = controlnet(
            x=z.float(),
            timesteps=timestep,
            context=context,
            controlnet_cond=controlnet_condition.float()
        )
        noise_pred, _ = diffusion(
            x=z.float(),
            timesteps=timestep,
            context=context.float(),
            down_block_additional_residuals=down_h,
            mid_block_additional_residual=mid_h
        )
        if scheduler_name in ('DPM-Solver++', 'UniPC'):
            result = scheduler.step(noise_pred, t, z)
            z = result.prev_sample
        else:
            z, _ = scheduler.step(noise_pred, t, z)

    z = (z / scale_factor).sum(dim=0) / average_over_n
    return z


def main():
    # ---------- Configuration ----------
    input_num = 3
    scale_factor = 1.0
    diffusion_seed = 2025
    input_T = True
    multi_heads = True

    B = 1
    D, H, W = 48, 256, 512

    # ================== PHASE 1: ENCODING ==================
    print("Phase 1: Encoding...")
    autoencoder = wasserstein_autoencoder.init_wasserstein_autoencoder_inter(
        "./output_dummy/swae_best.pth",
        map_location='cpu'
    ).to(device).eval()

    source_np = np.random.randn(B, input_num, 1, D, H, W).astype(np.float32)
    target_np = np.random.randn(B, 1, D, H, W).astype(np.float32)
    starting_a_np = np.random.rand(B, input_num, 7).astype(np.float32)
    context_np = np.random.rand(B, 1, 7).astype(np.float32)

    source_tmp_all = []
    for ch in range(input_num):
        source_tmp = torch.from_numpy(source_np[:, ch, :, :, :]).to(device)
        with torch.no_grad():
            starting_z_tmp = autoencoder.encode(source_tmp)
        print(f"After Phase 1 encode: {torch.cuda.memory_allocated()/1e6:.2f} MB")
        source_tmp_all.append(starting_z_tmp.cpu())

        # Delete GPU tensors immediately to free memory
        del starting_z_tmp
        del source_tmp

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Concatenate latents (all on CPU)
    starting_z = torch.cat(source_tmp_all, dim=1)
    del source_tmp_all

    starting_a = torch.from_numpy(starting_a_np)
    context = torch.from_numpy(context_np)

    del autoencoder, source_np, starting_a_np, context_np
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()

    print(f"After Phase 1 cleanup: {torch.cuda.memory_allocated()/1e6:.2f} MB")

    # ================== PHASE 2: SAMPLING ==================
    print("Phase 2: Sampling...")
    # Load diffusion and controlnet to CPU first, then move to GPU
    diffusion = climb.init_latent_diffusion_4(
        "./dummy_diffusion_output/diffusion_best.pth",
        map_location='cpu',
        multi_heads=multi_heads
    ).to(device).eval()

    controlnet = climb.init_controlnet_4(
        "./controlnet_dummy_output/controlnet_best.pth",
        imput_time_num=input_num,
        map_location='cpu',
        input_T=input_T,
        multi_heads=multi_heads
    ).to(device).eval()

    temporal_fuser = None

    # Move data to GPU for sampling
    starting_z_gpu = starting_z.to(device)
    starting_a_gpu = starting_a.to(device)
    context_gpu = context.to(device)

    z = sample_latent_using_controlnet(
        diffusion=diffusion,
        controlnet=controlnet,
        starting_z=starting_z_gpu,
        starting_a=starting_a_gpu,
        context=context_gpu,
        device=device,
        scale_factor=scale_factor,
        average_over_n=10,
        num_inference_steps=25,
        scheduler_name='DDIM',
        verbose=True,
        seed=diffusion_seed,
        input_T=input_T
    )

    # Release models and GPU data immediately after sampling
    del diffusion, controlnet, temporal_fuser, starting_z_gpu, starting_a_gpu, context_gpu
    torch.cuda.empty_cache()
    gc.collect()
    print(f"After Phase 2: {torch.cuda.memory_allocated()/1e6:.2f} MB")
    print("Phase 2 complete. Diffusion and ControlNet unloaded.")

    # ================== PHASE 3: DECODING ==================
    print("Phase 3: Decoding...")
    # Reload autoencoder (CPU first then GPU)
    autoencoder = wasserstein_autoencoder.init_wasserstein_autoencoder_inter(
        "./output_dummy/swae_best.pth",
        map_location='cpu'
    ).to(device).eval()

    # Remove padding if present
    if z.shape[1] == 8:
        z_cropped = z[:, 1:-1, :, :]
    else:
        z_cropped = z

    # Decode
    predicted_image = autoencoder.decode(z_cropped.unsqueeze(0))

    predicted_image_np = predicted_image.detach().cpu().numpy()[0, 0]

    # Post-processing
    predicted_image_np[predicted_image_np < 0.0025] = 0
    predicted_image_np[predicted_image_np < 0] = 0
    predicted_image_np[predicted_image_np > 1] = 1

    predicted_processed = get_largest_connected_component(predicted_image_np)
    target_processed = get_largest_connected_component(target_np[0, 0])

    print("Inference completed successfully.")
    print(f"Target processed shape: {target_processed.shape}")
    print(f"Predicted processed shape: {predicted_processed.shape}")

    del autoencoder, predicted_image
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()