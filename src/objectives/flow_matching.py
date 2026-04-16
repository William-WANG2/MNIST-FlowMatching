from abc import ABC, abstractmethod

import torch


def _validate_model_output_shape(output: torch.Tensor, target: torch.Tensor) -> None:
    if output.shape != target.shape:
        raise ValueError(
            f"Model output shape {tuple(output.shape)} does not match target shape "
            f"{tuple(target.shape)}."
        )


class FlowMatchingObjective(ABC):
    def __init__(self, probability_path, eps: float):
        self.probability_path = probability_path
        self.eps = eps

    def sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self.eps + (1.0 - self.eps) * torch.rand(batch_size, device=device)

    @abstractmethod
    def compute_loss(
        self,
        model,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Return the scalar training loss for one batch."""


class UnconditionalFlowMatchingObjective(FlowMatchingObjective):
    def compute_loss(
        self,
        model,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = images.shape[0]
        t = self.sample_time(batch_size, images.device)
        x_t = self.probability_path.sample_path(images, t)
        target = self.probability_path.vector_field(x_t, images, t)
        output = model(x_t, t, None)
        _validate_model_output_shape(output, target)
        return torch.mean((output - target) ** 2)

class CFGFlowMatchingObjective(FlowMatchingObjective):
    def __init__(
        self,
        probability_path,
        null_label: int,
        label_dropout: float,
        eps: float,
    ):
        super().__init__(probability_path=probability_path, eps=eps)
        self.null_label = null_label
        self.label_dropout = label_dropout

    def compute_loss(
        self,
        model,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = images.shape[0]
        t = self.sample_time(batch_size, images.device)
        x_t = self.probability_path.sample_path(images, t)
        target = self.probability_path.vector_field(x_t, images, t)
        # Apply classifier-free guidance label dropout
        dropout_mask = torch.rand(batch_size, device=images.device) < self.label_dropout
        labels = torch.where(dropout_mask, self.null_label, labels)
        output = model(x_t, t, labels)
        _validate_model_output_shape(output, target)
        return torch.mean((output - target) ** 2)
