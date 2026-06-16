from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def bandwidth_ratio_to_channels(ratio: float) -> int:
    """Return the CIFAR-10 bottleneck width C for k/n = ratio.

    The encoder produces 8 * 8 * C real values, representing k complex
    channel uses. Since n = 32 * 32 * 3, k/n = C/96.
    """
    channels = round(96 * ratio)
    if channels < 1 or not math.isclose(channels / 96, ratio, abs_tol=1e-8):
        raise ValueError(f"ratio={ratio} cannot be represented exactly; use C/96")
    return channels


class Encoder(nn.Module):
    def __init__(self, bottleneck_channels: int) -> None:
        super().__init__()
        widths = [3, 16, 32, 32, 32, bottleneck_channels]
        strides = [2, 2, 1, 1, 1]
        layers: list[nn.Module] = []
        for in_channels, out_channels, stride in zip(widths, widths[1:], strides):
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=5,
                        stride=stride,
                        padding=2,
                    ),
                    nn.PReLU(out_channels),
                ]
            )
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self, bottleneck_channels: int) -> None:
        super().__init__()
        widths = [bottleneck_channels, 32, 32, 32, 16, 3]
        strides = [1, 1, 1, 2, 2]
        layers: list[nn.Module] = []
        for index, (in_channels, out_channels, stride) in enumerate(
            zip(widths, widths[1:], strides)
        ):
            output_padding = 1 if stride == 2 else 0
            layers.append(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=5,
                    stride=stride,
                    padding=2,
                    output_padding=output_padding,
                )
            )
            layers.append(nn.Sigmoid() if index == 4 else nn.PReLU(out_channels))
        self.layers = nn.Sequential(*layers)

    def forward(self, z: Tensor) -> Tensor:
        return self.layers(z)


class PowerNormalization(nn.Module):
    """Enforce ||z||^2 = kP independently for every image."""

    def __init__(self, power: float = 1.0, eps: float = 1e-8) -> None:
        super().__init__()
        self.power = power
        self.eps = eps

    def forward(self, z_real: Tensor) -> Tensor:
        flat = z_real.flatten(1)
        complex_symbols = flat.shape[1] / 2
        target_norm = math.sqrt(complex_symbols * self.power)
        norm = flat.square().sum(dim=1, keepdim=True).clamp_min(self.eps).sqrt()
        return (flat * (target_norm / norm)).view_as(z_real)


class WirelessChannel(nn.Module):
    """Differentiable complex AWGN or slow Rayleigh fading channel."""

    def __init__(self, kind: str = "awgn", power: float = 1.0) -> None:
        super().__init__()
        if kind not in {"awgn", "rayleigh"}:
            raise ValueError(f"unsupported channel: {kind}")
        self.kind = kind
        self.power = power

    @staticmethod
    def _split_complex(z_real: Tensor) -> tuple[Tensor, Tensor]:
        flat = z_real.flatten(1)
        if flat.shape[1] % 2:
            raise ValueError("the bottleneck must contain 2k real values")
        return flat.chunk(2, dim=1)

    def forward(self, z_real: Tensor, snr_db: float | Tensor) -> Tensor:
        snr = torch.as_tensor(snr_db, dtype=z_real.dtype, device=z_real.device)
        noise_power = self.power / torch.pow(10.0, snr / 10.0)
        component_std = torch.sqrt(noise_power / 2.0)

        if self.kind == "rayleigh":
            real, imag = self._split_complex(z_real)
            shape = (z_real.shape[0], 1)
            h_real = torch.randn(shape, device=z_real.device, dtype=z_real.dtype) / math.sqrt(2)
            h_imag = torch.randn(shape, device=z_real.device, dtype=z_real.dtype) / math.sqrt(2)
            faded_real = h_real * real - h_imag * imag
            faded_imag = h_real * imag + h_imag * real
            z_real = torch.cat([faded_real, faded_imag], dim=1).view_as(z_real)

        return z_real + torch.randn_like(z_real) * component_std


class DeepJSCC(nn.Module):
    def __init__(
        self,
        bandwidth_ratio: float = 1 / 12,
        channel: str = "awgn",
        power: float = 1.0,
    ) -> None:
        super().__init__()
        self.bandwidth_ratio = bandwidth_ratio
        self.bottleneck_channels = bandwidth_ratio_to_channels(bandwidth_ratio)
        self.encoder = Encoder(self.bottleneck_channels)
        self.normalize = PowerNormalization(power)
        self.channel = WirelessChannel(channel, power)
        self.decoder = Decoder(self.bottleneck_channels)

    def encode(self, x: Tensor) -> Tensor:
        return self.normalize(self.encoder(x))

    def forward(self, x: Tensor, snr_db: float | Tensor) -> Tensor:
        z = self.encode(x)
        z_hat = self.channel(z, snr_db)
        return self.decoder(z_hat)
