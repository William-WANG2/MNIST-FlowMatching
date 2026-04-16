from pathlib import Path

from omegaconf import OmegaConf

from data.mnist import build_train_loader
from engine.latent_flow_trainer import LatentFlowTrainer
from engine.trainer import Trainer
from engine.vae_trainer import VAETrainer
from factories import (
    build_backbone,
    build_latent_objective,
    build_objective,
    build_probability_path,
    build_representation_shape,
    build_vae_model,
)
from latent_vae import load_frozen_vae
from objectives.vae import VAEObjective
from utils.runtime import ensure_dir, resolve_device, seed_everything


def run_training(cfg, project_root: Path) -> None:
    seed_everything(int(cfg.training.seed))
    device = resolve_device(cfg.training.device)

    task_name = getattr(cfg.task, "name", "flow_matching")
    representation_name = getattr(cfg.representation, "name", "pixel")
    if task_name == "flow_matching":
        default_run_name = (
            f"{cfg.backbone.name}_{cfg.conditioning.name}_{cfg.simulator.name}"
            if representation_name == "pixel"
            else f"{cfg.backbone.name}_{cfg.conditioning.name}_{cfg.simulator.name}_{representation_name}"
        )
    else:
        default_run_name = task_name
    run_name = cfg.training.run_name or default_run_name
    run_dir = ensure_dir(project_root / cfg.training.output_root / run_name)

    OmegaConf.save(cfg, run_dir / "config.yaml")

    train_loader = build_train_loader(
        data_root=project_root / cfg.data.root,
        batch_size=int(cfg.training.batch_size),
        image_size=int(cfg.data.image_size),
        mean=float(cfg.data.mean),
        std=float(cfg.data.std),
        num_workers=int(cfg.data.num_workers),
        pin_memory=bool(cfg.data.pin_memory),
    )

    if task_name == "flow_matching":
        sample_shape = build_representation_shape(cfg)
        probability_path = build_probability_path(
            cfg,
            image_shape=sample_shape,
        )
        model = build_backbone(cfg, image_shape=sample_shape).to(device)

        trainer_kwargs = {}
        if representation_name == "latent_vae":
            vae = load_frozen_vae(cfg, project_root=project_root, device=device)
            objective = build_latent_objective(cfg, probability_path, vae=vae)
            trainer_cls = LatentFlowTrainer
            trainer_kwargs.update(
                vae=vae,
                sample_posterior=bool(cfg.representation.sample_posterior),
                latent_shape=sample_shape,
            )
        else:
            objective = build_objective(cfg, probability_path)
            trainer_cls = Trainer
    elif task_name == "vae":
        model = build_vae_model(cfg).to(device)
        objective = VAEObjective()
        trainer_cls = VAETrainer
        trainer_kwargs = {}
    else:
        raise ValueError(f"Unsupported training task: {task_name}")

    trainer = trainer_cls(
        model=model,
        objective=objective,
        dataloader=train_loader,
        device=device,
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
        warmup_steps=int(cfg.training.warmup_steps),
        checkpoint_every=int(cfg.training.checkpoint_every),
        log_every=int(cfg.training.log_every),
        image_log_every=int(cfg.training.image_log_every),
        loss_curve_every=int(cfg.training.loss_curve_every),
        compare_batch_size=int(cfg.training.compare_batch_size),
        run_dir=run_dir,
        **trainer_kwargs,
    )
    trainer.train(num_steps=int(cfg.training.num_steps))
