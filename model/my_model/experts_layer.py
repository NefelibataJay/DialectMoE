from typing import Tuple, Any

import torch
import torch.nn as nn


class Expert(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, dropout_rate: float, ):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x) -> torch.Tensor:
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return self.dropout(out)


class ExpertsLayer(nn.Module):
    def __init__(self,
                 d_model: int = 256,
                 expert_dim: int = 1024,
                 n_experts: int = 8,
                 dropout_rate: float = 0.1,
                 capacity_factor: float = 1.25,
                 drop_tokens: bool = True,
                 is_scale_prob: bool = True,
                 ):
        super().__init__()
        self.capacity_factor = capacity_factor
        self.is_scale_prob = is_scale_prob
        self.n_experts = n_experts
        self.drop_tokens = drop_tokens

        self.experts = torch.nn.ModuleList(
            [Expert(input_size=d_model, output_size=d_model, hidden_size=expert_dim) for _ in range(n_experts)])

        self.switch = torch.nn.Linear(d_model, n_experts)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, ) -> Tuple[Any, Any, Any, int, Any]:
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
        # [(batch_size * seq_len), n_experts]  对应位置就是当前这个语音片段交给哪个专家的概率
        route_prob = self.softmax(self.switch(x))

        # 求出最大概率，和对应专家
        # -> torch.topk(route_prob,k=1, dim=-1)
        route_prob_max, routes = torch.max(route_prob, dim=-1)

        # 计算整个输入中那一部分归哪一部分专家负责
        indexes_list = [torch.eq(routes, i).nonzero().squeeze() for i in range(self.n_experts)]

        final_output = x.new_zeros(x.shape)

        capacity = int(self.capacity_factor * len(x) / self.n_experts)

        # 计算当前专家所接受的token数目
        counts = x.new_tensor([len(indexes_list[i]) for i in range(self.n_experts)])

        dropped = []

        if self.drop_tokens:
            for i in range(self.n_experts):
                if len(indexes_list[i]) <= capacity:
                    continue
                # 打乱顺序
                indexes_list[i] = indexes_list[i][torch.randperm(len(indexes_list[i]))]
                # 抽取最后 3 段， 保留前capacity个token
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
