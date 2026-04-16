import torch

from engine.trainer import Trainer
from latent_vae import decode_with_vae
from visualization import save_training_comparison


class LatentFlowTrainer(Trainer):
    def __init__(self, vae, sample_posterior: bool, latent_shape, **kwargs):
        super().__init__(**kwargs)
        self.vae = vae
        self.sample_posterior = sample_posterior
        self.latent_shape = tuple(latent_shape)

    @torch.no_grad()
    def _save_training_comparison(
        self,
        step: int,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        batch_size = min(self.compare_batch_size, images.shape[0])
        target_images = images[:batch_size]
        target_labels = labels[:batch_size] if labels is not None else None

        target_latents = self.objective.encode_batch(target_images)
        t = self.objective.sample_time(batch_size=batch_size, device=target_latents.device)
        x_t = self.objective.probability_path.sample_path(target_latents, t)
        velocity = self.model(x_t, t, target_labels)
        t_view = t.view(-1, *([1] * (x_t.ndim - 1)))
        predicted_latents = x_t + (1.0 - t_view) * velocity

        decoded_noisy = decode_with_vae(self.vae, x_t)
        decoded_predicted = decode_with_vae(self.vae, predicted_latents)

        log_path = self.run_dir / "train_logs" / "comparisons" / f"step_{step:06d}.png"
        save_training_comparison(
            targets=target_images,
            noisy_inputs=decoded_noisy,
            predictions=decoded_predicted,
            output_path=log_path,
            nrow=max(1, min(8, batch_size)),
        )
