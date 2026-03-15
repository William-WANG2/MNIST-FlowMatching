from models.cnn import CNNVectorField
from models.dit import DiTVectorField
from models.mlp import MLPVectorField
from objectives.flow_matching import (
    CFGFlowMatchingObjective,
    UnconditionalFlowMatchingObjective,
)
from probability_paths.gaussian import (
    GaussianProbabilityPath,
    LinearAlpha,
    LinearBeta,
)
from simulation.sampler import FlowSampler


def build_probability_path(cfg, image_shape):
    if cfg.path.name != "gaussian":
        raise ValueError(f"Unsupported probability path: {cfg.path.name}")
    if cfg.path.schedule.alpha != "linear" or cfg.path.schedule.beta != "linear":
        raise ValueError("Only linear alpha/beta schedules are configured in this repo.")
    return GaussianProbabilityPath(
        sample_shape=image_shape,
        source_std=float(cfg.path.source_std),
        alpha=LinearAlpha(),
        beta=LinearBeta(),
    )


def build_backbone(cfg, image_shape):
    conditioning_classes = None
    if cfg.conditioning.enabled:
        conditioning_classes = int(cfg.conditioning.num_model_labels)
        expected_null_label = conditioning_classes - 1
        if int(cfg.conditioning.null_label) != expected_null_label:
            raise ValueError(
                "CFG expects the last label slot to be the unconditional/null label."
            )
        if conditioning_classes != int(cfg.data.num_classes) + 1:
            raise ValueError(
                "CFG expects one extra label slot beyond the MNIST digit labels."
            )
    if cfg.backbone.name == "mlp":
        return MLPVectorField(
            image_shape=image_shape,
            hidden_dims=list(cfg.backbone.hidden_dims),
            time_embed_dim=int(cfg.backbone.time_embed_dim),
            class_embed_dim=int(cfg.backbone.class_embed_dim),
            num_classes=conditioning_classes,
        )
    if cfg.backbone.name == "cnn":
        return CNNVectorField(
            image_shape=image_shape,
            channels=list(cfg.backbone.channels),
            num_residual_layers=int(cfg.backbone.num_residual_layers),
            time_embed_dim=int(cfg.backbone.time_embed_dim),
            class_embed_dim=int(cfg.backbone.class_embed_dim),
            num_classes=conditioning_classes,
        )
    if cfg.backbone.name == "dit":
        return DiTVectorField(
            image_shape=image_shape,
            patch_size=int(cfg.backbone.patch_size),
            depth=int(cfg.backbone.depth),
            dim=int(cfg.backbone.dim),
            heads=int(cfg.backbone.heads),
            time_embed_dim=int(cfg.backbone.time_embed_dim),
            class_embed_dim=int(cfg.backbone.class_embed_dim),
            num_classes=conditioning_classes,
        )
    raise ValueError(f"Unsupported backbone: {cfg.backbone.name}")


def build_objective(cfg, probability_path):
    if cfg.conditioning.name == "cfg":
        return CFGFlowMatchingObjective(
            probability_path=probability_path,
            null_label=int(cfg.conditioning.null_label),
            label_dropout=float(cfg.conditioning.training_dropout),
            eps=float(cfg.path.eps),
        )
    if cfg.conditioning.name == "none":
        return UnconditionalFlowMatchingObjective(
            probability_path=probability_path,
            eps=float(cfg.path.eps),
        )
    raise ValueError(f"Unsupported conditioning mode: {cfg.conditioning.name}")


def build_sampler(cfg, probability_path):
    return FlowSampler(
        probability_path=probability_path,
        simulator_name=cfg.simulator.name,
        t_start=float(cfg.simulator.t_start),
        t_end=float(cfg.simulator.t_end),
        num_steps=int(cfg.simulator.num_steps),
        conditioning_enabled=bool(cfg.conditioning.enabled),
        null_label=None if cfg.conditioning.null_label is None else int(cfg.conditioning.null_label),
        diffusion_scale=float(cfg.simulator.diffusion_scale),
    )
