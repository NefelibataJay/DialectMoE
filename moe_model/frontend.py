from typing import Tuple, Union, Optional, Any
import torch.nn
from wenet.utils.mask import make_pad_mask


class Conv2dSubsampling4(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
    """

    def __init__(self, idim: int, odim: int):
        """Construct an Conv2dSubsampling4 object."""
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim))
        # The right context for every conv layer is computed by:
        # (kernel_size - 1) * frame_rate_of_this_layer
        self.subsampling_rate = 4
        # 6 = (3 - 1) * 1 + (3 - 1) * 2
        self.right_context = 6

    def forward(
            self,
            x: torch.Tensor,
            x_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.unsqueeze(1)  # (b, c=1, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        return x, x_mask[:, :, 2::2][:, :, 2::2]


class Frontend(torch.nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int = 256,
                 input_layer: Optional[str] = "conv2d",
                 dropout_rate: float = 0.0,
                 global_cmvn: torch.nn.Module = None,
                 ):
        super().__init__()
        self.global_cmvn = global_cmvn
        if input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(input_size, output_size),
                torch.nn.LayerNorm(output_size),
                torch.nn.Dropout(dropout_rate),
            )
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling4(
                input_size,
                output_size,
            )
        else:
            raise ValueError("unknown input_layer: " + input_layer)

    def forward(self, x: torch.Tensor, ilens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        T = x.size(1)
        masks = ~make_pad_mask(ilens, T).unsqueeze(1)  # (B, 1, T)
        if self.global_cmvn is not None:
            xs = self.global_cmvn(x)
        x, x_mask = self.embed(x, masks)
        return x, x_mask
