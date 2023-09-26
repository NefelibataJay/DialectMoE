from typing import Tuple, Any, Optional

import torch
import torch.nn as nn

from loss.balance_loss import SparseL1Loss, BalanceImportanceLoss


class Adapter(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, dropout_rate: float = 0.0, ):
        super(Adapter, self).__init__()
        self.w_1 = nn.Linear(input_size, hidden_size)
        self.w_2 = nn.Linear(hidden_size, output_size)
        # self.activate = nn.ReLU()
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class AdaptersFeedForward(nn.Module):
    def __init__(self,
                 input_size: int = 256,
                 embed_dim: int = 256,
                 adapter_dim: int = 1024,
                 num_adapter: int = 4,
                 dropout_rate: float = 0.1,
                 capacity_factor: float = 1.5,
                 drop_tokens: bool = True,
                 is_scale_prob: bool = True,
                 route_type: str = 'none',
                 ):
        super().__init__()
        assert route_type in ["concat", "add", "embed", "none"]
        if route_type == "concat":
            router_input_dim = input_size + embed_dim
        elif route_type == "add":
            router_input_dim = input_size
        elif route_type == "embed":
            router_input_dim = embed_dim
        else:
            # none
            router_input_dim = input_size

        self.n_adapters = num_adapter
        self.drop_tokens = drop_tokens
        self.capacity_factor = capacity_factor
        self.route_type = route_type
        self.is_scale_prob = is_scale_prob

        self.experts = torch.nn.ModuleList([Adapter(input_size=input_size,
                                                   output_size=input_size,
                                                   hidden_size=adapter_dim,
                                                   dropout_rate=dropout_rate) for _ in range(self.n_adapters)])

        self.router = torch.nn.Linear(router_input_dim, self.n_adapters)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout_rate)

        self.sparseLoss = SparseL1Loss()
        self.balanceLoss = BalanceImportanceLoss()

    def forward(self, x: torch.Tensor, embed: Optional[torch.Tensor] = None,)-> torch.Tensor:
        """
        Args:
            x: input tensor of shape (batch_size, seq_len, d_model)
            embed: domain prompt tensor of shape (batch_size, seq_len, d_model)
        """
        if embed is not None:
            if self.route_type == "concat":
                router_inputs = torch.cat([embed, x], dim=-1)
            elif self.route_type == "add":
                router_inputs = embed + x
            elif self.route_type == "embed":
                router_inputs = embed
            else:
                router_inputs = x
        else:
            router_inputs = x

        router_logits = self.router(router_inputs)
        router_probs = self.softmax(router_logits)

        gate_value, gate_idx = router_probs.max(dim=-1)

        indexes_list = [torch.eq(gate_idx, i).nonzero().squeeze(1) for i in range(self.n_adapters)]
        
        output = x.new_zeros(x.shape)
        expert_output = []

        if self.is_scale_prob:
            final_output = final_output * gate_value.view(-1, 1)
        else:
            final_output = final_output * (gate_value / gate_value.detach()).view(-1, 1)

        return output
