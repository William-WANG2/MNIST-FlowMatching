from latent_vae import expected_latent_shape
from models.cnn import CNNVectorField
from models.dit import DiTVectorField
from models.mlp import MLPVectorField
from models.vae import VAE
from objectives.flow_matching import (
    CFGFlowMatchingObjective,
    UnconditionalFlowMatchingObjective,
)
from objectives.latent_flow_matching import LatentFlowMatchingObjective
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


def build_representation_shape(cfg):
    representation_name = getattr(cfg.representation, "name", "pixel")
    if representation_name == "pixel":
        return (
            int(cfg.data.channels),
            int(cfg.data.image_size),
            int(cfg.data.image_size),
        )
    if representation_name == "latent_vae":
        return expected_latent_shape(cfg)
    raise ValueError(f"Unsupported representation: {representation_name}")


def build_backbone(cfg, image_shape):
    representation_name = getattr(cfg.representation, "name", "pixel")
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
        hidden_dims = getattr(cfg.backbone, "hidden_dims", None)
        if hidden_dims is None:
            hidden_dims = (
                cfg.backbone.latent_hidden_dims
                if representation_name == "latent_vae"
                else cfg.backbone.pixel_hidden_dims
            )
        return MLPVectorField(
            image_shape=image_shape,
            hidden_dims=list(hidden_dims),
            time_embed_dim=int(cfg.backbone.time_embed_dim),
            class_embed_dim=int(cfg.backbone.class_embed_dim),
            num_classes=conditioning_classes,
        )
    if cfg.backbone.name == "cnn":
        channels = getattr(cfg.backbone, "channels", None)
        if channels is None:
            channels = (
                cfg.backbone.latent_channels
                if representation_name == "latent_vae"
                else cfg.backbone.pixel_channels
            )
        num_residual_layers = getattr(cfg.backbone, "num_residual_layers", None)
        if num_residual_layers is None:
            num_residual_layers = (
                cfg.backbone.latent_num_residual_layers
                if representation_name == "latent_vae"
                else cfg.backbone.pixel_num_residual_layers
            )
        mid_num_residual_layers = getattr(cfg.backbone, "mid_num_residual_layers", None)
        if mid_num_residual_layers is None:
            mid_num_residual_layers = (
                cfg.backbone.latent_mid_num_residual_layers
                if representation_name == "latent_vae"
                else cfg.backbone.pixel_mid_num_residual_layers
            )

        downsample = getattr(cfg.backbone, "downsample", None)
        if downsample is None:
            downsample = (
                cfg.backbone.latent_downsample
                if representation_name == "latent_vae"
                else cfg.backbone.pixel_downsample
            )

        upsample_mode = getattr(cfg.backbone, "upsample_mode", None)
        if upsample_mode is None:
            upsample_mode = (
                cfg.backbone.latent_upsample_mode
                if representation_name == "latent_vae"
                else cfg.backbone.pixel_upsample_mode
            )

        return CNNVectorField(
            image_shape=image_shape,
            channels=list(channels),
            num_residual_layers=int(num_residual_layers),
            mid_num_residual_layers=None
            if mid_num_residual_layers is None
            else int(mid_num_residual_layers),
            downsample=bool(downsample),
            upsample_mode=str(upsample_mode),
            time_embed_dim=int(cfg.backbone.time_embed_dim),
            class_embed_dim=int(cfg.backbone.class_embed_dim),
            num_classes=conditioning_classes,
        )
    if cfg.backbone.name == "dit":
        patch_size = getattr(cfg.backbone, "patch_size", None)
        if patch_size is None:
            patch_size = (
                cfg.backbone.latent_patch_size
                if representation_name == "latent_vae"
                else cfg.backbone.pixel_patch_size
            )
        return DiTVectorField(
            image_shape=image_shape,
            patch_size=int(patch_size),
            depth=int(cfg.backbone.depth),
            dim=int(cfg.backbone.dim),
            heads=int(cfg.backbone.heads),
            time_embed_dim=int(cfg.backbone.time_embed_dim),
            class_embed_dim=int(cfg.backbone.class_embed_dim),
            num_classes=conditioning_classes,
        )
    raise ValueError(f"Unsupported backbone: {cfg.backbone.name}")


def build_vae_model(cfg):
    return VAE(
        data_channels=int(cfg.data.channels),
        hidden_channels=list(cfg.vae.hidden_channels),
        beta=float(cfg.vae.beta),
    )


def build_objective(cfg, probability_path):
    representation_name = getattr(cfg.representation, "name", "pixel")
    if representation_name == "latent_vae":
        raise ValueError("Use build_objective(cfg, probability_path, vae=...) for latent VAE runs.")
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


def build_latent_objective(cfg, probability_path, vae):
    if cfg.conditioning.name == "cfg":
        null_label = int(cfg.conditioning.null_label)
        label_dropout = float(cfg.conditioning.training_dropout)
    elif cfg.conditioning.name == "none":
        null_label = None
        label_dropout = 0.0
    else:
        raise ValueError(f"Unsupported conditioning mode: {cfg.conditioning.name}")

    return LatentFlowMatchingObjective(
        probability_path=probability_path,
        vae=vae,
        sample_posterior=bool(cfg.representation.sample_posterior),
        latent_shape=expected_latent_shape(cfg),
        null_label=null_label,
        label_dropout=label_dropout,
        eps=float(cfg.path.eps),
    )


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
