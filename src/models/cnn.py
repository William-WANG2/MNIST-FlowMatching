from typing import Optional

import torch
from torch import nn

from models.base import VectorFieldModel
from models.embeddings import FourierEncoder


class ResidualLayer(nn.Module):
    def __init__(
        self,
        channels: int,
        time_embed_dim: int,
        class_embed_dim: Optional[int],
    ):
        super().__init__()
        norm_groups = _resolve_group_norm_groups(channels)
        cond_dim = time_embed_dim + (class_embed_dim or 0)

        self.norm1 = nn.GroupNorm(norm_groups, channels)
        self.norm2 = nn.GroupNorm(norm_groups, channels)
        self.activation = nn.SiLU()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.condition_adapter = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, channels * 2),
        )
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(
        self,
        x: torch.Tensor,
        t_embed: torch.Tensor,
        y_embed: Optional[torch.Tensor],
    ) -> torch.Tensor:
        residual = x
        conditioning = t_embed if y_embed is None else torch.cat([t_embed, y_embed], dim=-1)
        scale, bias = self.condition_adapter(conditioning).chunk(2, dim=-1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        bias = bias.unsqueeze(-1).unsqueeze(-1)

        x = self.norm1(x)
        x = x * (1.0 + scale) + bias
        x = self.activation(x)
        x = self.conv1(x)
        x = self.activation(self.norm2(x))
        x = self.conv2(x)
        return residual + x


def _resolve_group_norm_groups(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class Encoder(nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        num_residual_layers: int,
        time_embed_dim: int,
        class_embed_dim: Optional[int],
        downsample: bool,
    ):
        super().__init__()
        self.res_blocks = nn.ModuleList(
            [
                ResidualLayer(channels_in, time_embed_dim, class_embed_dim)
                for _ in range(num_residual_layers)
            ]
        )
        stride = 2 if downsample else 1
        self.transition = nn.Conv2d(
            channels_in,
            channels_out,
            kernel_size=3,
            stride=stride,
            padding=1,
        )

    def forward(
        self,
        x: torch.Tensor,
        t_embed: torch.Tensor,
        y_embed: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for block in self.res_blocks:
            x = block(x, t_embed, y_embed)
        skip = x
        x = self.transition(x)
        return x, skip


class Midcoder(nn.Module):
    def __init__(
        self,
        channels: int,
        num_residual_layers: int,
        time_embed_dim: int,
        class_embed_dim: Optional[int],
    ):
        super().__init__()
        self.res_blocks = nn.ModuleList(
            [
                ResidualLayer(channels, time_embed_dim, class_embed_dim)
                for _ in range(num_residual_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        t_embed: torch.Tensor,
        y_embed: Optional[torch.Tensor],
    ) -> torch.Tensor:
        for block in self.res_blocks:
            x = block(x, t_embed, y_embed)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        num_residual_layers: int,
        time_embed_dim: int,
        class_embed_dim: Optional[int],
        upsample: bool,
        upsample_mode: str,
    ):
        super().__init__()
        if upsample_mode not in {"bilinear", "deconv"}:
            raise ValueError(
                f"Unsupported upsample_mode: {upsample_mode}. "
                "Expected one of {'bilinear', 'deconv'}."
            )

        if upsample:
            if upsample_mode == "deconv":
                self.upsample = nn.ConvTranspose2d(
                    channels_in,
                    channels_out,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            else:
                self.upsample = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1),
                )
        else:
            self.upsample = nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1)
        self.res_blocks = nn.ModuleList(
            [
                ResidualLayer(channels_out, time_embed_dim, class_embed_dim)
                for _ in range(num_residual_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        t_embed: torch.Tensor,
        y_embed: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x = self.upsample(x)
        x = x + skip
        for block in self.res_blocks:
            x = block(x, t_embed, y_embed)
        return x


class CNNVectorField(VectorFieldModel):
    def __init__(
        self,
        image_shape,
        channels,
        num_residual_layers: int,
        mid_num_residual_layers: Optional[int],
        downsample: bool,
        upsample_mode: str,
        time_embed_dim: int,
        class_embed_dim: int,
        num_classes: Optional[int],
    ):
        super().__init__(image_shape=image_shape, num_classes=num_classes)
        image_channels = image_shape[0]
        effective_class_embed_dim = class_embed_dim if num_classes is not None else None

        self.null_label = num_classes - 1 if num_classes is not None else None
        self.time_embedder = FourierEncoder(time_embed_dim)
        self.time_projection = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.class_embedding = (
            nn.Embedding(num_classes, class_embed_dim) if num_classes is not None else None
        )
        self.class_projection = (
            nn.Sequential(
                nn.Linear(class_embed_dim, class_embed_dim),
                nn.SiLU(),
                nn.Linear(class_embed_dim, class_embed_dim),
            )
            if num_classes is not None
            else None
        )
        if upsample_mode not in {"bilinear", "deconv"}:
            raise ValueError(
                f"Unsupported upsample_mode: {upsample_mode}. "
                "Expected one of {'bilinear', 'deconv'}."
            )
        if mid_num_residual_layers is None:
            mid_num_residual_layers = num_residual_layers

        self.init_conv = nn.Sequential(
            nn.Conv2d(image_channels, channels[0], kernel_size=3, padding=1),
            nn.GroupNorm(_resolve_group_norm_groups(channels[0]), channels[0]),
            nn.SiLU(),
        )
        self.encoders = nn.ModuleList(
            [
                Encoder(
                    channels_in=current_channels,
                    channels_out=next_channels,
                    num_residual_layers=num_residual_layers,
                    time_embed_dim=time_embed_dim,
                    class_embed_dim=effective_class_embed_dim,
                    downsample=downsample,
                )
                for current_channels, next_channels in zip(channels[:-1], channels[1:])
            ]
        )
        self.midcoder = Midcoder(
            channels=channels[-1],
            num_residual_layers=mid_num_residual_layers,
            time_embed_dim=time_embed_dim,
            class_embed_dim=effective_class_embed_dim,
        )
        self.decoders = nn.ModuleList(
            list(
                reversed(
                    [
                        Decoder(
                            channels_in=next_channels,
                            channels_out=current_channels,
                            num_residual_layers=num_residual_layers,
                            time_embed_dim=time_embed_dim,
                            class_embed_dim=effective_class_embed_dim,
                            upsample=downsample,
                            upsample_mode=upsample_mode,
                        )
                        for current_channels, next_channels in zip(channels[:-1], channels[1:])
                    ]
                )
            )
        )
        self.final_conv = nn.Conv2d(
            channels[0],
            image_channels,
            kernel_size=3,
            padding=1,
        )
        self.final_norm = nn.GroupNorm(
            _resolve_group_norm_groups(channels[0]),
            channels[0],
        )
        self.final_activation = nn.SiLU()
        nn.init.zeros_(self.final_conv.weight)
        nn.init.zeros_(self.final_conv.bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        t_embed = self.time_projection(self.time_embedder(t))
        y_embed = None
        if self.class_embedding is not None:
            if y is None:
                raise ValueError("Conditional CNNVectorField requires labels.")
            y_embed = self.class_projection(self.class_embedding(y))

        x = self.init_conv(x)
        input_skip = x

        skips = []
        for encoder in self.encoders:
            x, skip = encoder(x, t_embed, y_embed)
            skips.append(skip)

        x = self.midcoder(x, t_embed, y_embed)

        for decoder in self.decoders:
            x = decoder(x, skips.pop(), t_embed, y_embed)

        x = x + input_skip
        x = self.final_activation(self.final_norm(x))
        return self.final_conv(x)
