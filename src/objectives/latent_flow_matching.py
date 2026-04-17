import torch

from latent_vae import encode_with_vae
from objectives.flow_matching import FlowMatchingObjective, _validate_model_output_shape


class LatentFlowMatchingObjective(FlowMatchingObjective):
    def __init__(
        self,
        probability_path,
        vae,
        sample_posterior: bool,
        latent_shape,
        null_label: int | None,
        label_dropout: float,
        eps: float,
    ):
        super().__init__(probability_path=probability_path, eps=eps)
        self.vae = vae
        self.sample_posterior = sample_posterior
        self.latent_shape = tuple(latent_shape)
        self.null_label = null_label
        self.label_dropout = label_dropout

    @torch.no_grad()
    def encode_batch(self, images: torch.Tensor) -> torch.Tensor:
        return encode_with_vae(
            vae=self.vae,
            images=images,
            sample_posterior=self.sample_posterior,
            latent_shape=self.latent_shape,
        )

    def compute_loss(
        self,
        model,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        latents = self.encode_batch(images)
        batch_size = latents.shape[0]
        t = self.sample_time(batch_size, latents.device)
        x_t, source = self.probability_path.sample_coupled(latents, t)
        target = self.probability_path.vector_field_from_source(latents, source, t)

        effective_labels = None
        if self.null_label is not None:
            dropout_mask = torch.rand(batch_size, device=latents.device) < self.label_dropout
            effective_labels = torch.where(dropout_mask, self.null_label, labels)

        output = model(x_t, t, effective_labels)
        _validate_model_output_shape(output, target)
        return torch.mean((output - target) ** 2)
