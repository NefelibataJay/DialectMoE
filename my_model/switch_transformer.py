import torch
import torch.nn as nn
from torch.nn import Module

from wenet.transformer.attention import MultiHeadedAttention


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.soft = nn.Softmax(1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.soft(out)
        return out


class SwitchFeedForward(Module):
    def __init__(self,
                 capacity_factor: float,
                 drop_tokens: bool,
                 is_scale_prob: bool,
                 n_experts: int,
                 expert: Module,
                 d_model: int, ) -> None:

        super().__init__()
        self.capacity_factor = capacity_factor
        self.is_scale_prob = is_scale_prob
        self.n_experts = n_experts
        self.drop_tokens = drop_tokens

        self.experts = torch.nn.ModuleList([expert for _ in range(n_experts)])
        self.switch = torch.nn.Linear(d_model, n_experts)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor):
        """
        return:
            the final output
            number of tokens routed to each expert
            sum of probabilities for each expert
            number of tokens dropped.
            routing probabilities of the selected experts
        """
        batch_size, seq_len, d_model = x.shape

        x = x.view(-1, d_model)

        route_prob = self.softmax(self.switch(x))

        route_prob_max, routes = torch.max(route_prob, dim=-1)

        # 计算整个输入中那一部分归哪一部分专家负责
        indexes_list = [torch.eq(routes, i).nonzero(as_tuple=True)[0] for i in range(self.n_experts)]

        final_output = x.new_zeros(x.shape)

        capacity = int(self.capacity_factor * len(x) / self.n_experts)

        counts = x.new_tensor([len(indexes_list[i]) for i in range(self.n_experts)])

        dropped = []

        if self.drop_tokens:
            for i in range(self.n_experts):
                if len(indexes_list[i]) <= capacity:
                    continue
                indexes_list[i] = indexes_list[i][torch.randperm(len(indexes_list[i]))]

                dropped.append(indexes_list[i][capacity:])

                indexes_list[i] = indexes_list[i][:capacity]

        expert_output = [self.experts[i](x[indexes_list[i], :]) for i in range(self.n_experts)]

        for i in range(self.n_experts):
            final_output[indexes_list[i], :] = expert_output[i]

        if dropped:
            dropped = torch.cat(dropped)
            final_output[dropped, :] = x[dropped, :]

        if self.is_scale_prob:
            final_output = final_output * route_prob_max.view(-1, 1)
        else:
            final_output = final_output * (route_prob_max / route_prob_max.detach()).view(-1, 1)

        final_output = final_output.view(batch_size, seq_len, d_model)

        return final_output, counts, route_prob.sum(0), len(dropped), route_prob_max


class SwitchTransformerLayer(Module):
    def __init__(self,
                 d_model: int,
                 attention: MultiHeadedAttention,
                 feed_forward: SwitchFeedForward,
                 dropout_prob: float):
        super().__init__()
        self.size = d_model

        self.attention = attention
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.feed_forward = feed_forward
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.dropout = torch.nn.Dropout(dropout_prob)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        # multi-head attention
        x = self.norm1(x)
        x, _ = self.attention(x, x, x, mask=mask)
        x = x + self.dropout(x)

        # feed forward
        x = self.norm2(x)
        ff, counts, route_prob, n_dropped, route_prob_max = self.feed_forward(x)
        x = x + self.dropout(ff)

        return x, counts, route_prob, n_dropped, route_prob_max


class SwitchTransformer(Module):
    def __init__(self,
                 d_model: int = 256,
                 num_layers: int = 6,
                 capacity_factor: float = 1.0,
                 drop_tokens: bool = False,
                 is_scale_prob: bool = True,
                 n_experts: int = 8,
                 expert_dim: int = 1024,
                 att_dropout_rate: float = 0.0,
                 dropout_rate: float = 0.1
                 ):
        """
        Args:
            capacity_factor is the capacity of each expert as a factor relative to ideally balanced load
            drop_tokens specifies whether to drop tokens if more tokens are routed to an expert than the capacity
            is_scale_prob specifies whether to multiply the input to the FFN by the routing probability
            n_experts is the number of experts
            expert is the expert layer, a FFN module
            d_model model dim
        """
        super().__init__()

        expert = MLP(d_model, d_model, expert_dim)

        self.layers = torch.nn.ModuleList([SwitchTransformerLayer(
            d_model=d_model,
            attention=MultiHeadedAttention(n_head=8, n_feat=d_model, dropout_rate=att_dropout_rate),
            feed_forward=SwitchFeedForward(capacity_factor=capacity_factor, drop_tokens=drop_tokens,
                                           is_scale_prob=is_scale_prob, n_experts=n_experts, expert=expert,
                                           d_model=d_model),
            dropout_prob=dropout_rate
        ) for _ in range(num_layers)])

        self.norm = torch.nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        counts = []
        route_probs = []
        n_dropped = []
        route_prob_max = []
        for layer in self.layers:
            x, f, p, n_d, p_max = layer(x, mask)
            counts.append(f)
            route_probs.append(p)
            n_dropped.append(n_d)
            route_prob_max.append(p_max)

        x = self.norm(x)

        return x, torch.stack(counts), torch.stack(route_probs), n_dropped, torch.stack(route_prob_max)
