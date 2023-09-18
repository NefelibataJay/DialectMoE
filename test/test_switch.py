import torch

from my_model.switch_transformer import SwitchTransformer


def test_switch():
    load_balancing_loss_ceof = 0.01
    model = SwitchTransformer(d_model=16,capacity_factor=2.0)

    x = torch.randn(8, 3, 16)
    mask = torch.ones(8, 3, 3)

    x, counts, route_probs, n_dropped, route_prob_max = model(x, mask)
    total = counts.sum(dim=-1, keepdims=True)
    route_frac = counts / total
    route_prob = route_probs / total
    load_balancing_loss = 8 * (route_frac * route_prob).sum()

    experts_loss = load_balancing_loss_ceof * load_balancing_loss




if __name__ == "__main__":
    model = SwitchTransformer()

    x = torch.randn(2, 3, 256)
    mask = torch.ones(2, 3, 3)

    x, counts, route_probs, n_dropped, route_prob_max = model(x, mask)
