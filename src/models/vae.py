from typing import Optional

from models.dit import MultiHeadSelfAttention
import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn


class ResidualBlock(nn.Module):
    """ Two applications of LN + convolution + non-linearity + residual connection """
    def __init__(self, channels: int, act: nn.Module = nn.SiLU):
        super().__init__()

        # Init norm, convolutions, and activations
        self.norm = nn.GroupNorm(1, channels)
        
        # 3x3 convolution (padding=1 is required to preserve height and width for the residual connection)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        
        # Instantiate the activation function passed as a parameter
        self.act = act()
        
        # 1x1 convolution
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1)

        # Initialize the second convolution to zero - stabilizes training early on!
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor):
        # Res init
        x_skip = x

        # Norm
        x = self.norm(x)

        # First convolution
        x = self.conv1(x)

        # Activation
        x = self.act(x)

        # Second convolution
        x = self.conv2(x)

        # Return residual connection
        return x_skip + x


class AttnBlock(nn.Module):
    """Self-attention over flattened spatial tokens."""

    def __init__(self, channels: int):
        super().__init__()
        self.reshape1 = Rearrange("b c h w -> b (h w) c")

        self.norm1 = nn.LayerNorm(channels)
        self.att = MultiHeadSelfAttention(channels, heads=4)
        self.norm2 = nn.LayerNorm(channels)
        self.ff = nn.Sequential(
            nn.Linear(channels, 4 * channels),
            nn.SiLU(),
            nn.Linear(4 * channels, channels)
        )
        nn.init.zeros_(self.ff[-1].weight)
        nn.init.zeros_(self.ff[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        x = self.reshape1(x)
        attn_skip = x
        x = self.norm1(x)
        x = self.att(x)
        x = x + attn_skip
        ff_skip = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + ff_skip

        return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class EncoderBlock(nn.Module):
    """Residual/attention block with optional spatial downsampling."""

    def __init__(self, in_channels: int, downsample_channels: Optional[int] = None):
        super().__init__()

        self.res1 = ResidualBlock(in_channels)
        self.res2 = ResidualBlock(in_channels)
        self.attn = AttnBlock(in_channels)
        if downsample_channels is not None:
            self.downsample = nn.Conv2d(
                in_channels=in_channels,
                out_channels=downsample_channels,
                kernel_size=3,
                padding=1,
                stride=2,
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.res1(x)
        x = self.res2(x)
        x = self.attn(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: list[int]):
        super().__init__()
        self.init_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels[0],
            kernel_size=3,
            padding=1,
            stride=1,
        )

        ch_in = hidden_channels
        ch_out = hidden_channels[1:] + [None]
        self.blocks = nn.ModuleList(
            [EncoderBlock(in_c, out_c) for in_c, out_c in zip(ch_in, ch_out)]
        )

        z_dim = hidden_channels[-1]
        # Both mean and log-variance are predicted from the input so that the encoder
        # can express different levels of uncertainty for different inputs.
        self.z_mean = nn.Sequential(
            nn.GroupNorm(1, z_dim),
            nn.Conv2d(in_channels=z_dim, out_channels=z_dim, kernel_size=1, stride=1, padding=0),
        )
        self.z_logvar = nn.Sequential(
            nn.GroupNorm(1, z_dim),
            nn.Conv2d(in_channels=z_dim, out_channels=z_dim, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Initial convolution
        x = self.init_conv(x)

        # Propagate through encoder blocks
        for block in self.blocks:
            x = block(x)

        # Predict input-conditioned mean and log-variance
        z_mean = self.z_mean(x)
        z_logvar = self.z_logvar(x).clamp(-30.0, 20.0)
        return z_mean, z_logvar


class DecoderBlock(nn.Module):
    """Residual/attention block with optional spatial upsampling."""

    def __init__(self, in_channels: int, upsample_channels: Optional[int] = None):
        super().__init__()
        self.res1 = ResidualBlock(in_channels)
        self.res2 = ResidualBlock(in_channels)
        self.attn = AttnBlock(in_channels)
        if upsample_channels is not None:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=upsample_channels,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                ),
            )
        else:
            self.upsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.res1(x)
        x = self.res2(x)
        x = self.attn(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class Decoder(nn.Module):
    def __init__(self, out_channels: int, hidden_channels: list[int]):
        super().__init__()

        ch_in = hidden_channels
        ch_out = hidden_channels[1:] + [None]
        self.blocks = nn.ModuleList(
            [DecoderBlock(in_c, out_c) for in_c, out_c in zip(ch_in, ch_out)]
        )

        x_dim = hidden_channels[-1]
        self.x_mean = nn.Sequential(
            nn.GroupNorm(1, x_dim),
            nn.Conv2d(
                in_channels=x_dim,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )
        # Per-channel log-variance (broadcasts over spatial dims) — more expressive
        # than a single global scalar.
        self.logvar = nn.Parameter(torch.zeros(out_channels, 1, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            x = block(x)
        x_mean = self.x_mean(x)
        # Clamp to prevent logvar → -∞ (which sends NLL → -∞).
        # Range [-30, 20] follows the Stable Diffusion / latent-diffusion convention.
        x_logvar = self.logvar.clamp(-30.0, 20.0)
        return x_mean, x_logvar


class VAE(nn.Module):
    '''Variational autoencoder with Gaussian latent distribution and diagonal covariance.'''
    def __init__(self, data_channels: int, hidden_channels: list[int], beta: float = 0.1):
        super().__init__()
        self.beta = beta
        self._encoder = Encoder(data_channels, hidden_channels)
        self._decoder = Decoder(data_channels, list(reversed(hidden_channels)))

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._encoder(x)

    def decode(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_mean, z_logvar = self.encode(x)

        # Reparameterization is wired here so the trainer can remain plug-and-play.
        z = z_mean + torch.exp(0.5 * z_logvar) * torch.randn_like(z_mean)
        x_mean, x_logvar = self.decode(z)
        return z_mean, z_logvar, x_mean, x_logvar

    def compute_loss(
        self,
        z_mean: torch.Tensor,
        z_logvar: torch.Tensor,
        x_mean: torch.Tensor,
        x_logvar: torch.Tensor,
        x_true: torch.Tensor,
    ) -> torch.Tensor:
        """See the VAE objective derived in the lab notes."""

        batch_size = z_mean.shape[0]

        # KL loss: -0.5 * sum(1 + log(var) - mean^2 - var)
        # Sum over all latent dimensions per sample, then average over the batch.
        # Using .mean() here would divide by (z_dim * H * W) instead of just the batch
        # size, making the effective beta depend on the latent tensor size and causing
        # extreme sensitivity to architecture choices.
        kl_div = -0.5 * (1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
        kl_loss = self.beta * kl_div.view(batch_size, -1).sum(dim=1).mean()

        # Reconstruction loss: Gaussian Negative Log-Likelihood
        # 0.5 * sum( (x_true - x_mean)^2 / exp(x_logvar) + x_logvar )
        # Sum over all data dimensions per sample, then average over the batch.
        nll = 0.5 * (torch.exp(-x_logvar) * (x_true - x_mean).pow(2) + x_logvar)
        recon_loss = nll.view(batch_size, -1).sum(dim=1).mean()

        return kl_loss + recon_loss
