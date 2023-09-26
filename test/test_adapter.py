import torch
import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from model.moduel import adapter_feedforward


def test_experts():
    inp = torch.randn(5, 3, 2)
    embed = torch.randn(5, 3, 2)
    eff = adapter_feedforward.AdaptersFeedForward(input_size=2, embed_dim=2, adapter_dim=5, num_adapter=4, dropout_rate=0.1,)
    final_output = eff(inp, embed)
    print(final_output.shape)

    # experts = ExpertsLayer(d_model=2, expert_dim=5, n_experts=4,)
    # final_output, counts, route_prob, n_dropped, route_prob_max = experts(inp)
    # total = counts.sum(dim=-1, keepdims=True)
    # route_frac = counts / total
    # route_prob = route_prob / total
    # load_balancing_loss = 8 * (route_frac * route_prob).sum()
    #
    # experts_loss = load_balancing_loss
    # print(final_output.shape)

if __name__ == '__main__':
    test_experts()
