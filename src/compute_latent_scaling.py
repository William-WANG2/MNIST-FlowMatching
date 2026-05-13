"""Compute the latent scaling factor for a trained VAE.

Usage:
    python compute_latent_scaling.py vae_checkpoint_path=runs/vae/checkpoint_final.pt

Encodes the full training set with the VAE, computes std(z), and prints the
scaling factor (1/std) to set in configs/representation/latent_vae.yaml.
"""

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from data.mnist import build_train_loader
from factories import build_vae_model
from latent_vae import encode_with_vae, infer_vae_latent_shape, resolve_checkpoint_path
from utils.runtime import resolve_device


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    project_root = Path(__file__).resolve().parent.parent

    checkpoint_path = getattr(cfg, "vae_checkpoint_path", None)
    if checkpoint_path is None:
        checkpoint_path = getattr(cfg.representation, "vae_checkpoint_path", None)
    if checkpoint_path is None:
        raise ValueError(
            "Provide vae_checkpoint_path=<path> on the command line or set "
            "representation.vae_checkpoint_path in the config."
        )

    device = resolve_device("auto")
    resolved_path = resolve_checkpoint_path(checkpoint_path, project_root)

    vae = build_vae_model(cfg).to(device)
    state_dict = torch.load(resolved_path, map_location=device)
    vae.load_state_dict(state_dict)
    vae.eval()

    latent_shape = infer_vae_latent_shape(cfg)
    loader = build_train_loader(
        data_root=project_root / cfg.data.root,
        batch_size=256,
        image_size=int(cfg.data.image_size),
        mean=float(cfg.data.mean),
        std=float(cfg.data.std),
        num_workers=int(cfg.data.num_workers),
        pin_memory=False,
    )

    print("Encoding training set...")
    sum_sq = torch.zeros(1, device=device)
    sum_val = torch.zeros(1, device=device)
    count = 0

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            z = encode_with_vae(vae, images, sample_posterior=False, latent_shape=latent_shape)
            sum_val += z.sum()
            sum_sq += z.pow(2).sum()
            count += z.numel()

    mean = (sum_val / count).item()
    var = (sum_sq / count).item() - mean ** 2
    std = var ** 0.5
    scaling_factor = 1.0 / std

    print(f"\nLatent statistics over training set:")
    print(f"  mean = {mean:.6f}")
    print(f"  std  = {std:.6f}")
    print(f"  scaling_factor (1/std) = {scaling_factor:.6f}")
    print(f"\nSet in configs/representation/latent_vae.yaml:")
    print(f"  latent_scaling_factor: {scaling_factor:.4f}")


if __name__ == "__main__":
    main()
