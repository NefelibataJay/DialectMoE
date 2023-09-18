import torch

from my_model.encoder import MyEncoder
from wenet.utils.mask import make_pad_mask


def test_mymodel():
    encoder = MyEncoder()
    x = torch.randn(8, 11, 256)
    x_len = torch.tensor([11, 10, 8, 10, 9, 1, 3, 6])
    masks = ~make_pad_mask(x_len, x.size(1)).unsqueeze(1)  # (B, 1, T)
    encoder_out, encoder_mask, info = encoder(x, x_len, masks)

    counts = info[0]
    route_probs = info[1]
    total = counts.sum(dim=-1, keepdims=True)
    route_frac = counts / total
    route_prob = route_probs / total
    balancing_loss = 8 * (route_frac * route_prob).sum()
    print("ok")

