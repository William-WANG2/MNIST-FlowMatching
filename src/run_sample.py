from pathlib import Path

import torch
from omegaconf import OmegaConf

from factories import build_backbone, build_probability_path, build_sampler
from utils.runtime import ensure_dir, resolve_device, seed_everything
from visualization import save_image_grid


def run_sampling(cfg, project_root: Path) -> None:
    if cfg.sampling.checkpoint_path is None:
        raise ValueError("Set sampling.checkpoint_path to a trained checkpoint before sampling.")

    seed_everything(0)
    device = resolve_device("auto")
    output_dir = ensure_dir(project_root / cfg.sampling.output_dir)

    probability_path = build_probability_path(
        cfg,
        image_shape=(cfg.data.channels, cfg.data.image_size, cfg.data.image_size),
    )
    model = build_backbone(
        cfg,
        image_shape=(cfg.data.channels, cfg.data.image_size, cfg.data.image_size),
    ).to(device)
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
        save_image_grid(
            samples=samples,
            output_path=output_dir / "samples_unconditional.png",
            nrow=int(cfg.sampling.samples_per_class),
        )
