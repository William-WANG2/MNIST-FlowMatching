from pathlib import Path

from omegaconf import OmegaConf

from data.mnist import build_train_loader
from engine.trainer import Trainer
from factories import build_backbone, build_objective, build_probability_path
from utils.runtime import ensure_dir, resolve_device, seed_everything


def run_training(cfg, project_root: Path) -> None:
    seed_everything(int(cfg.training.seed))
    device = resolve_device(cfg.training.device)

    run_name = cfg.training.run_name or (
        f"{cfg.backbone.name}_{cfg.conditioning.name}_{cfg.simulator.name}"
    )
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
    probability_path = build_probability_path(
        cfg,
        image_shape=(cfg.data.channels, cfg.data.image_size, cfg.data.image_size),
    )
    model = build_backbone(
        cfg,
        image_shape=(cfg.data.channels, cfg.data.image_size, cfg.data.image_size),
    ).to(device)
    objective = build_objective(cfg, probability_path)

    trainer = Trainer(
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
    )
    trainer.train(num_steps=int(cfg.training.num_steps))
