from pathlib import Path

import torch
from omegaconf import OmegaConf

from factories import (
    build_backbone,
    build_probability_path,
    build_representation_shape,
    build_sampler,
    build_vae_model,
)
from latent_vae import decode_with_vae, infer_vae_latent_shape, load_frozen_vae
from utils.runtime import ensure_dir, resolve_device, seed_everything
from visualization import save_image_grid


def run_sampling(cfg, project_root: Path) -> None:
    if cfg.sampling.checkpoint_path is None:
        raise ValueError("Set sampling.checkpoint_path to a trained checkpoint before sampling.")

    seed_everything(0)
    device = resolve_device("auto")
    output_dir = ensure_dir(project_root / cfg.sampling.output_dir)
    task_name = getattr(cfg.task, "name", "flow_matching")
    if task_name == "vae":
        model = build_vae_model(cfg).to(device)
        state_dict = torch.load(cfg.sampling.checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        latent_shape = infer_vae_latent_shape(cfg)
        batch_size = int(cfg.sampling.samples_per_class) * int(cfg.data.num_classes)
        latents = torch.randn(batch_size, *latent_shape, device=device)
        with torch.no_grad():
            samples = decode_with_vae(model, latents)

        OmegaConf.save(cfg, output_dir / "sample_config.yaml")
        save_image_grid(
            samples=samples,
            output_path=output_dir / "samples_vae.png",
            nrow=int(cfg.sampling.samples_per_class),
        )
        return

    representation_name = getattr(cfg.representation, "name", "pixel")
    sample_shape = build_representation_shape(cfg)
    vae = None
    if representation_name == "latent_vae":
        vae = load_frozen_vae(cfg, project_root=project_root, device=device)

    probability_path = build_probability_path(
        cfg,
        image_shape=sample_shape,
    )
    model = build_backbone(cfg, image_shape=sample_shape).to(device)
    state_dict = torch.load(cfg.sampling.checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    sampler = build_sampler(cfg, probability_path)
    OmegaConf.save(cfg, output_dir / "sample_config.yaml")

    if cfg.conditioning.enabled:
        labels = torch.arange(int(cfg.data.num_classes), device=device).repeat_interleave(
            int(cfg.sampling.samples_per_class)
        )
        for guidance_scale in cfg.sampling.guidance_scales:
            samples = sampler.sample(
                model=model,
                batch_size=labels.shape[0],
                device=device,
                labels=labels,
                guidance_scale=float(guidance_scale),
                use_tqdm=bool(cfg.sampling.use_tqdm),
            )
            if vae is not None:
                samples = decode_with_vae(vae, samples)
            save_image_grid(
                samples=samples,
                output_path=output_dir / f"samples_guidance_{guidance_scale:.1f}.png",
                nrow=int(cfg.sampling.samples_per_class),
            )
    else:
        batch_size = int(cfg.sampling.samples_per_class) * int(cfg.data.num_classes)
        samples = sampler.sample(
            model=model,
            batch_size=batch_size,
            device=device,
            labels=None,
            guidance_scale=1.0,
            use_tqdm=bool(cfg.sampling.use_tqdm),
        )
        if vae is not None:
            samples = decode_with_vae(vae, samples)
        save_image_grid(
            samples=samples,
            output_path=output_dir / "samples_unconditional.png",
            nrow=int(cfg.sampling.samples_per_class),
        )
