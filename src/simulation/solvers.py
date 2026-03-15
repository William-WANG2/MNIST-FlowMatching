from abc import ABC, abstractmethod

import torch
from tqdm import tqdm


class ODE(ABC):
    @abstractmethod
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """Return the ODE drift."""


class SDE(ABC):
    @abstractmethod
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """Return the SDE drift."""

    @abstractmethod
    def diffusion_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """Return the SDE diffusion coefficient."""


class Solver(ABC):
    @abstractmethod
    def step(self, xt: torch.Tensor, t: torch.Tensor, dt: torch.Tensor, **kwargs) -> torch.Tensor:
        """Advance the state by one time step."""

    @torch.no_grad()
    def simulate(self, x: torch.Tensor, ts: torch.Tensor, use_tqdm: bool = True, **kwargs) -> torch.Tensor:
        iterator = tqdm(range(ts.shape[1] - 1), disable=not use_tqdm)
        for index in iterator:
            t = ts[:, index]
            dt = ts[:, index + 1] - ts[:, index]
            x = self.step(x, t, dt, **kwargs)
        return x


class EulerSolver(Solver):
    def __init__(self, ode: ODE):
        self.ode = ode

    def step(self, xt: torch.Tensor, t: torch.Tensor, dt: torch.Tensor, **kwargs) -> torch.Tensor:
        step_size = dt.view((dt.shape[0],) + (1,) * (xt.ndim - 1))
        return xt + self.ode.drift_coefficient(xt, t, **kwargs) * step_size


class EulerMaruyamaSolver(Solver):
    def __init__(self, sde: SDE):
        self.sde = sde

    def step(self, xt: torch.Tensor, t: torch.Tensor, dt: torch.Tensor, **kwargs) -> torch.Tensor:
        step_size = dt.view((dt.shape[0],) + (1,) * (xt.ndim - 1))
        noise = torch.randn_like(xt)
        drift = self.sde.drift_coefficient(xt, t, **kwargs) * step_size
        diffusion = self.sde.diffusion_coefficient(xt, t, **kwargs) * torch.sqrt(step_size) * noise
        return xt + drift + diffusion


def build_time_grid(
    batch_size: int,
    num_steps: int,
    t_start: float,
    t_end: float,
    device: torch.device,
) -> torch.Tensor:
    return torch.linspace(t_start, t_end, num_steps, device=device).view(1, -1).expand(batch_size, -1)

