from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import nn


class VectorFieldModel(nn.Module, ABC):
    def __init__(self, image_shape, num_classes: Optional[int]):
        super().__init__()
        self.image_shape = tuple(image_shape)
        self.num_classes = num_classes

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict the vector field at time `t`."""

