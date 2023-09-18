import torch

from my_model.experts_layer import ExpertsLayer
from moe_model.experts_feedforward import ExpertsFeedForward


def test_experts():
    inp = torch.randn(5, 3, 2)
    embed = torch.randn(5, 3, 2)
    eff = ExpertsFeedForward(input_size=2, embed_dim=2, expert_dim=5, num_experts=4, dropout_rate=0.1,)
    final_output, loss = eff(inp, embed)
    print(loss)

    # experts = ExpertsLayer(d_model=2, expert_dim=5, n_experts=4,)
    # final_output, counts, route_prob, n_dropped, route_prob_max = experts(inp)
    # total = counts.sum(dim=-1, keepdims=True)
    # route_frac = counts / total
    # route_prob = route_prob / total
    # load_balancing_loss = 8 * (route_frac * route_prob).sum()
    #
    # experts_loss = load_balancing_loss
    # print(final_output.shape)
