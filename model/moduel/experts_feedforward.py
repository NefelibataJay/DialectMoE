from typing import Tuple, Any, Optional

import torch
import torch.nn as nn

from loss.balance_loss import SparseL1Loss, BalanceImportanceLoss


class Expert(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, dropout_rate: float = 0.0, ):
        super(Expert, self).__init__()
        self.w_1 = nn.Linear(input_size, hidden_size)
        self.w_2 = nn.Linear(hidden_size, output_size)
        # self.activate = nn.ReLU()
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class ExpertsFeedForward(nn.Module):
    def __init__(self,
                 input_size: int = 256,
                 embed_dim: int = 256,
                 expert_dim: int = 1024,
                 num_experts: int = 4,
                 dropout_rate: float = 0.1,
                 capacity_factor: float = 1.5,
                 drop_tokens: bool = True,
                 is_scale_prob: bool = True,
                 fusion_type: str = 'none',
                 attention_router: bool = True,
                 ):
        super().__init__()
        assert fusion_type in ["concat", "add", "embed", "none"]
        if fusion_type == "concat":
            router_input_dim = input_size + embed_dim
        elif fusion_type == "add":
            router_input_dim = input_size
        elif fusion_type == "embed":
            router_input_dim = embed_dim
        else:
            # none
            router_input_dim = input_size

        self.n_experts = num_experts
        self.drop_tokens = drop_tokens
        self.capacity_factor = capacity_factor
        self.fusion_type = fusion_type
        self.is_scale_prob = is_scale_prob

        self.experts = torch.nn.ModuleList([Expert(input_size=input_size,
                                                   output_size=input_size,
                                                   hidden_size=expert_dim,
                                                   dropout_rate=dropout_rate) for _ in range(num_experts)])

        self.router = torch.nn.Linear(router_input_dim, num_experts)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout_rate)

        self.sparseLoss = SparseL1Loss()
        self.balanceLoss = BalanceImportanceLoss()

    def forward(self, x: torch.Tensor, embed: Optional[torch.Tensor] = None,)-> Tuple[torch.Tensor, tuple]:
        """
        Args:
            x: input tensor of shape (batch_size, seq_len, d_model)
            embed: domain prompt tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, input_dim = x.size()
        x = x.view(-1, input_dim)
        if embed is not None:
            embed_dim = embed.size(-1)
            embed = embed.view(-1, embed_dim)
            if self.fusion_type == "concat":
                router_inputs = torch.cat([embed, x], dim=-1)
            elif self.fusion_type == "add":
                assert embed_dim == input_dim
                router_inputs = embed + x
            elif self.fusion_type == "embed":
                router_inputs = embed
            else:
                # none
                router_inputs = x
        else:
            router_inputs = x

        router_logits = self.router(router_inputs)
        router_probs = self.softmax(router_logits)
        gate_value, gate_idx = router_probs.max(dim=-1)
        l1_loss = self.sparseLoss(router_probs)
        importance_loss = self.balanceLoss(router_probs)
        aux_loss = (l1_loss, importance_loss)

        all_samples = router_inputs.size(0)

        capacity = int(self.capacity_factor * all_samples / self.n_experts)

        indexes_list = [torch.eq(gate_idx, i).nonzero().squeeze(1) for i in range(self.n_experts)]

        # 计算每个专家所接受的token数目
        # counts = x.new_tensor([len(indexes_list[i]) for i in range(self.n_experts)])

        # 计算当前专家所接受的token数目
        dropped = []
        final_output = x.new_zeros(x.shape)
        if self.drop_tokens:
            for i in range(self.n_experts):
                if len(indexes_list[i]) <= capacity:
                    continue
                # 打乱顺序
                indexes_list[i] = indexes_list[i][torch.randperm(len(indexes_list[i]))]
                # 抽取最后 3 段， 保留前capacity个token
                dropped.append(indexes_list[i][capacity:])
                indexes_list[i] = indexes_list[i][:capacity]

        expert_output = []
        for i in range(self.n_experts):
            expert_output.append(self.experts[i](x[indexes_list[i], :]))
        # expert_output = [self.experts[i](x[indexes_list[i], :]) for i in range(self.n_experts)]

        for i in range(self.n_experts):
            final_output[indexes_list[i], :] = expert_output[i]

        if dropped:
            dropped = torch.cat(dropped)
            final_output[dropped, :] = x[dropped, :]

        if self.is_scale_prob:
            final_output = final_output * gate_value.view(-1, 1)
        else:
            final_output = final_output * (gate_value / gate_value.detach()).view(-1, 1)

        final_output = final_output.view(batch_size, seq_len, input_dim)
        
        # total = counts.sum(dim=-1, keepdims=True)
        # route_frac = counts / total
        # route_probs = router_probs.sum(0) / total
        # load_balancing_loss = self.n_experts * (route_frac * route_probs).sum()

        return final_output, aux_loss
