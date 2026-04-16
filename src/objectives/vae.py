import torch


class VAEObjective:
    """Thin objective wrapper so VAE training fits the existing Trainer interface."""

    def compute_loss(
        self,
        model,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        del labels
        z_mean, z_logvar, x_mean, x_logvar = model(images)
        return model.compute_loss(z_mean, z_logvar, x_mean, x_logvar, images)
