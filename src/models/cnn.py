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
        self.block1 = nn.Sequential(
            nn.SiLU(),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.SiLU(),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.time_adapter = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, channels),
        )
        self.class_adapter = None
        if class_embed_dim is not None:
            self.class_adapter = nn.Sequential(
                nn.Linear(class_embed_dim, class_embed_dim),
                nn.SiLU(),
                nn.Linear(class_embed_dim, channels),
            )

    def forward(
        self,
        x: torch.Tensor,
        t_embed: torch.Tensor,
        y_embed: Optional[torch.Tensor],
    ) -> torch.Tensor:
        residual = x
        x = self.block1(x)
        x = x + self.time_adapter(t_embed).unsqueeze(-1).unsqueeze(-1)
        if self.class_adapter is not None and y_embed is not None:
            x = x + self.class_adapter(y_embed).unsqueeze(-1).unsqueeze(-1)
        x = self.block2(x)
        return x + residual


class Encoder(nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        num_residual_layers: int,
        time_embed_dim: int,
        class_embed_dim: Optional[int],
    ):
        super().__init__()
        self.res_blocks = nn.ModuleList(
            [
                ResidualLayer(channels_in, time_embed_dim, class_embed_dim)
                for _ in range(num_residual_layers)
            ]
        )
        self.downsample = nn.Conv2d(
            channels_in,
            channels_out,
            kernel_size=3,
            stride=2,
            padding=1,
        )

    def forward(
        self,
        x: torch.Tensor,
        t_embed: torch.Tensor,
        y_embed: Optional[torch.Tensor],
    ) -> torch.Tensor:
        for block in self.res_blocks:
            x = block(x, t_embed, y_embed)
        return self.downsample(x)


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
    ):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1),
        )
        self.res_blocks = nn.ModuleList(
            [
                ResidualLayer(channels_out, time_embed_dim, class_embed_dim)
                for _ in range(num_residual_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        t_embed: torch.Tensor,
        y_embed: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x = self.upsample(x)
        for block in self.res_blocks:
            x = block(x, t_embed, y_embed)
        return x


class CNNVectorField(VectorFieldModel):
    def __init__(
        self,
        image_shape,
        channels,
        num_residual_layers: int,
        time_embed_dim: int,
        class_embed_dim: int,
        num_classes: Optional[int],
    ):
        super().__init__(image_shape=image_shape, num_classes=num_classes)
        image_channels = image_shape[0]
        effective_class_embed_dim = class_embed_dim if num_classes is not None else None

        self.null_label = num_classes - 1 if num_classes is not None else None
        self.time_embedder = FourierEncoder(time_embed_dim)
        self.class_embedding = (
            nn.Embedding(num_classes, class_embed_dim) if num_classes is not None else None
        )
        self.init_conv = nn.Sequential(
            nn.Conv2d(image_channels, channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[0]),
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
                )
                for current_channels, next_channels in zip(channels[:-1], channels[1:])
            ]
        )
        self.midcoder = Midcoder(
            channels=channels[-1],
            num_residual_layers=num_residual_layers,
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

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        t_embed = self.time_embedder(t) # shape (batch_size, time_embed_dim)
        y_embed = self.class_embedding(y) if self.class_embedding is not None else None # shape (batch_size, class_embed_dim) or None

        x = self.init_conv(x) # shape (batch_size, channels[0], height, width)

        skips = []
        for encoder in self.encoders:
            x = encoder(x, t_embed, y_embed) # shape (batch_size, channels[i], height // 2, width // 2)
            skips.append(x)
        
        x = self.midcoder(x, t_embed, y_embed) # shape (batch_size, channels[-1], height // 2**len(channels), width // 2**len(channels))

        for decoder in self.decoders:
            skip = skips.pop() # shape (batch_size, channels[i], height // 2**i, width // 2**i)
            x = x + skip # Add skip connection before decoding. shape (batch_size, channels[i], height // 2**i, width // 2**i)
            x = decoder(x, t_embed, y_embed) # shape (batch_size, channels[i-1], height // 2**(i-1), width // 2**(i-1))
        x = self.final_conv(x) # shape (batch_size, image_channels, height, width)
        return x
