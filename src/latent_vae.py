from pathlib import Path
from typing import Sequence

import torch
from torch import nn


def resolve_checkpoint_path(checkpoint_path: str, project_root: Path) -> Path:
    path = Path(checkpoint_path)
    if not path.is_absolute():
        path = project_root / path
    return path


def expected_latent_shape(cfg) -> tuple[int, ...]:
    latent_shape = getattr(cfg.representation, "latent_shape", None)
    if latent_shape is None:
        raise ValueError("Set representation.latent_shape when using representation=latent_vae.")
    shape = tuple(int(dim) for dim in latent_shape)
    if len(shape) != 3:
        raise ValueError("representation.latent_shape must be a 3D shape like [C, H, W].")
    return shape


def infer_vae_latent_shape(cfg) -> tuple[int, int, int]:
    hidden_channels = list(cfg.vae.hidden_channels)
    if not hidden_channels:
        raise ValueError("cfg.vae.hidden_channels must contain at least one channel width.")

    image_size = int(cfg.data.image_size)
    downsample_factor = 2 ** max(0, len(hidden_channels) - 1)
    if image_size % downsample_factor != 0:
        raise ValueError(
            f"Image size {image_size} is incompatible with VAE downsample factor {downsample_factor}."
        )

    spatial_size = image_size // downsample_factor
    return int(hidden_channels[-1]), spatial_size, spatial_size


def encode_with_vae(
    vae: nn.Module,
    images: torch.Tensor,
    sample_posterior: bool,
    latent_shape: Sequence[int] | None = None,
) -> torch.Tensor:
    z_mean, z_logvar = vae.encode(images)
    latents = z_mean
    if sample_posterior:
        latents = z_mean + torch.exp(0.5 * z_logvar) * torch.randn_like(z_mean)
    if latent_shape is not None and tuple(latents.shape[1:]) != tuple(latent_shape):
        raise ValueError(
            f"Encoded latent shape {tuple(latents.shape[1:])} does not match configured "
            f"representation.latent_shape {tuple(latent_shape)}."
        )
    return latents


def decode_with_vae(vae: nn.Module, latents: torch.Tensor) -> torch.Tensor:
    x_mean, _ = vae.decode(latents)
    return x_mean


def load_frozen_vae(cfg, project_root: Path, device: torch.device) -> nn.Module:
    checkpoint_path = getattr(cfg.representation, "vae_checkpoint_path", None)
    if checkpoint_path is None:
        raise ValueError(
            "Set representation.vae_checkpoint_path when using representation=latent_vae."
        )

    from factories import build_vae_model

    resolved_path = resolve_checkpoint_path(checkpoint_path, project_root)
    vae = build_vae_model(cfg).to(device)
    state_dict = torch.load(resolved_path, map_location=device)
    vae.load_state_dict(state_dict)
    vae.eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    return vae
