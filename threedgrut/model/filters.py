import torch
import torch.nn as nn
import torch.nn.functional as F


class Filter(nn.Module):
    def __init__(
        self,
        filter_type: str | None = "none",
        iterations: int = 1,
        padding_mode: str = "replicate",
    ):
        super().__init__()
        if iterations < 0:
            raise ValueError(f"iterations must be non-negative, got {iterations}")

        self.filter_type = "none" if filter_type is None else str(filter_type).lower()
        self.iterations = int(iterations)
        self.padding_mode = padding_mode

        init_fn = getattr(self, f"_init_{self.filter_type}", None)
        forward_fn = getattr(self, f"_forward_{self.filter_type}", None)
        if forward_fn is None:
            raise ValueError(f"Unknown render.filter_type: {self.filter_type}")
        if init_fn is not None:
            init_fn()

    def _init_atrous(self) -> None:
        kernel = torch.tensor(
            [
                [1, 2, 1],
                [2, 4, 2],
                [1, 2, 1],
            ],
            dtype=torch.float32,
        )
        kernel = kernel / kernel.sum()
        self.register_buffer("kernel", kernel.view(1, 1, 3, 3), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return getattr(self, f"_forward_{self.filter_type}")(x)

    def _forward_none(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def _forward_atrous(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Filter expects [B, H, W, C], got {tuple(x.shape)}")
        if self.iterations == 0:
            return x

        y = x.permute(0, 3, 1, 2).contiguous()
        kernel = self.kernel.to(device=y.device, dtype=y.dtype).repeat(y.shape[1], 1, 1, 1)

        for i in range(self.iterations):
            dilation = 2 ** i
            y = F.pad(y, (dilation, dilation, dilation, dilation), mode=self.padding_mode)
            y = F.conv2d(
                y,
                kernel,
                dilation=dilation,
                groups=y.shape[1],
            )

        return y.permute(0, 2, 3, 1).contiguous()
