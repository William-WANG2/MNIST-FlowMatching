from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import nn


def _broadcast_time_like(values: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    view_shape = (values.shape[0],) + (1,) * (reference.ndim - 1)
    return values.view(view_shape)


class Alpha(ABC):
    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate alpha(t)."""

    @abstractmethod
    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate d/dt alpha(t)."""


class Beta(ABC):
    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate beta(t)."""

    @abstractmethod
    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate d/dt beta(t)."""


class LinearAlpha(Alpha):
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return t

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t)


class LinearBeta(Beta):
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return 1.0 - t

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        return -torch.ones_like(t)


class GaussianProbabilityPath(nn.Module):
    def __init__(self, sample_shape, source_std: float, alpha: Alpha, beta: Beta):
        super().__init__()
        self.sample_shape = tuple(sample_shape)
        self.source_std = source_std
        self.alpha = alpha
        self.beta = beta
        self.register_buffer("_device_anchor", torch.zeros(1), persistent=False)

    @property
    def device(self) -> torch.device:
        return self._device_anchor.device

    def sample_source(self, batch_size: int, device: torch.device | None = None) -> torch.Tensor:
        sample_device = device or self.device
        return self.source_std * torch.randn(batch_size, *self.sample_shape, device=sample_device)

    def sample_source_like(self, reference: torch.Tensor) -> torch.Tensor:
        return self.source_std * torch.randn_like(reference)

    def sample_path(
        self,
        target: torch.Tensor,
        t: torch.Tensor,
        source: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if source is None:
            source = self.sample_source_like(target)
        alpha_t = _broadcast_time_like(self.alpha(t), target)
        beta_t = _broadcast_time_like(self.beta(t), target)
        return alpha_t * target + beta_t * source

    def sample_coupled(self, target: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        source = self.sample_source_like(target)
        return self.sample_path(target=target, t=t, source=source), source

    def vector_field_from_source(
        self,
        target: torch.Tensor,
        source: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        dt_alpha_t = _broadcast_time_like(self.alpha.dt(t), target)
        dt_beta_t = _broadcast_time_like(self.beta.dt(t), target)
        return dt_alpha_t * target + dt_beta_t * source

    def vector_field(self, x: torch.Tensor, target: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        alpha_t = _broadcast_time_like(self.alpha(t), x)
        beta_t = _broadcast_time_like(self.beta(t), x)
        dt_alpha_t = _broadcast_time_like(self.alpha.dt(t), x)
        dt_beta_t = _broadcast_time_like(self.beta.dt(t), x)
        return (
            (dt_alpha_t - dt_beta_t * alpha_t / beta_t) * target
            + (dt_beta_t / beta_t) * x
        )

    def score(self, x: torch.Tensor, target: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        alpha_t = _broadcast_time_like(self.alpha(t), x)
        beta_t = _broadcast_time_like(self.beta(t), x)
        return (alpha_t * target - x) / ((self.source_std**2) * beta_t**2)
