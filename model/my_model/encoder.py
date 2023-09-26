import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union, Any

from torch import Tensor

from my_model.experts_layer import ExpertsLayer
from wenet.branchformer.cgmlp import ConvolutionalGatingMLP
from wenet.transformer.attention import (
    MultiHeadedAttention,
    RelPositionMultiHeadedAttention,
)
from wenet.transformer.embedding import (
    RelPositionalEncoding,
    PositionalEncoding,
    NoPositionalEncoding,
)



class MyEncoderLayer(nn.Module):
    def __init__(
            self,
            size: int,
            attn: Optional[torch.nn.Module],
            cgmlp: Optional[torch.nn.Module],
            experts: Optional[torch.nn.Module],
            dropout_rate: float,
            merge_method: str,
            cgmlp_weight: float = 0.5,
            attn_branch_drop_rate: float = 0.0,
    ):
        super().__init__()

        self.size = size
        self.attn = attn
        self.cgmlp = cgmlp
        self.merge_method = merge_method
        self.cgmlp_weight = cgmlp_weight
        self.attn_branch_drop_rate = attn_branch_drop_rate

        if attn is not None:
            self.norm_mha = nn.LayerNorm(size)  # for the MHA module
        if cgmlp is not None:
            self.norm_mlp = nn.LayerNorm(size)  # for the MLP module
        if experts is not None:
            self.norm_experts = nn.LayerNorm(size)  # for the experts module
        self.norm_final = nn.LayerNorm(size)  # for the final output of the block

        self.dropout = torch.nn.Dropout(dropout_rate)

        if self.merge_method == "concat":
            self.merge_proj = torch.nn.Linear(size + size, size)

        self.experts = experts

    def forward(
            self,
            x: torch.Tensor,
            mask: torch.Tensor,
            pos_emb: torch.Tensor,
            mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
            att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
            cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
    ) -> tuple[Any, Tensor, tuple]:
        """Compute encoded features.

        Args:
            x (Union[Tuple, torch.Tensor]): Input tensor  (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, time, time).
            pos_emb (torch.Tensor): positional encoding, must not be None
                for BranchformerEncoderLayer.
            mask_pad (torch.Tensor): batch padding mask used for conv module.
                (#batch, 1，time), (0, 0, 0) means fake mask.
            att_cache (torch.Tensor): Cache tensor of the KEY & VALUE
                (#batch=1, head, cache_t1, d_k * 2), head * d_k == size.
            cnn_cache (torch.Tensor): Convolution cache in cgmlp layer
                (#batch=1, size, cache_t2)

        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time, time.
            torch.Tensor: att_cache tensor,
                (#batch=1, head, cache_t1 + time, d_k * 2).
            torch.Tensor: cnn_cahce tensor (#batch, size, cache_t2).
        """
        stoch_layer_coeff = 1.0
        # Two branches
        x1 = x
        x2 = x

        # Branch 1: multi-headed attention module
        if self.attn is not None:
            x1 = self.norm_mha(x1)
            x_att, new_att_cache = self.attn(x1, x1, x1, mask, pos_emb, att_cache)
            x1 = self.dropout(x_att)

        # Branch 2: convolutional gating mlp
        # Fake new cnn cache here, and then change it in conv_module
        if self.cgmlp is not None:
            x2 = self.norm_mlp(x2)
            x2, new_cnn_cache = self.cgmlp(x2, mask_pad, cnn_cache)
            x2 = self.dropout(x2)

        # Merge two branches
        if self.merge_method == "concat":
            x = x + stoch_layer_coeff * self.dropout(
                self.merge_proj(torch.cat([x1, x2], dim=-1))
            )
        else:
            raise ValueError(f"Unsupported merge method: {self.merge_method}")

        if self.experts is not None:
            x = self.norm_experts(x)
            x, counts, route_prob, n_dropped, route_prob_max = self.experts(x)
            x = x + self.dropout(x)
            info = (counts, route_prob, n_dropped, route_prob_max)
        else:
            info = ()

        x = self.norm_final(x)

        return x, mask, info


class MyEncoder(nn.Module):
    def __init__(self,
                 output_size: int = 256,
                 num_blocks: int = 6,
                 attention_heads: int = 4,
                 attention_layer_type: str = "rel_selfattn",
                 pos_enc_layer_type: str = "rel_pos",
                 cgmlp_linear_units: int = 2048,
                 cgmlp_conv_kernel: int = 31,
                 use_linear_after_conv: bool = False,
                 gate_activation: str = "identity",
                 merge_method: str = "concat",
                 cgmlp_weight: Union[float, List[float]] = 0.5,
                 num_expert: int = 8,
                 expert_size: int = 1024,
                 capacity_factor: float = 1.25,
                 drop_tokens: bool = False,
                 is_scale_prob: bool = True,
                 dropout_rate: float = 0.1,
                 positional_dropout_rate: float = 0.1,
                 attention_dropout_rate: float = 0.0,
                 ):
        super().__init__()
        self._output_size = output_size

        if pos_enc_layer_type == "abs_pos":
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == "no_pos":
            pos_enc_class = NoPositionalEncoding
        elif pos_enc_layer_type == "rel_pos":
            assert attention_layer_type == "rel_selfattn"
            pos_enc_class = RelPositionalEncoding
        else:
            raise ValueError("unknown pos_enc_layer: " + pos_enc_layer_type)
        self.pos_enc = pos_enc_class(output_size, positional_dropout_rate)

        if attention_layer_type == "selfattn":
            encoder_selfattn_layer = MultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
            )
        elif attention_layer_type == "rel_selfattn":
            assert pos_enc_layer_type == "rel_pos"
            encoder_selfattn_layer = RelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
            )
        else:
            raise ValueError("unknown encoder_attn_layer: " + attention_layer_type)

        if isinstance(cgmlp_weight, float):
            cgmlp_weight = [cgmlp_weight] * num_blocks
        if len(cgmlp_weight) != num_blocks:
            raise ValueError(
                f"Length of cgmlp_weight ({len(cgmlp_weight)}) should be equal to "
                f"num_blocks ({num_blocks})"
            )

        cgmlp_layer = ConvolutionalGatingMLP
        cgmlp_layer_args = (
            output_size,
            cgmlp_linear_units,
            cgmlp_conv_kernel,
            dropout_rate,
            use_linear_after_conv,
            gate_activation,
        )

        self.encoders = nn.ModuleList([MyEncoderLayer(
            output_size,
            encoder_selfattn_layer(*encoder_selfattn_layer_args),
            cgmlp_layer(*cgmlp_layer_args),
            ExpertsLayer(
                output_size,
                expert_size,
                n_experts=num_expert,
                dropout_rate=dropout_rate,
                capacity_factor=capacity_factor,
                drop_tokens=drop_tokens,
                is_scale_prob=is_scale_prob,
            ),
            dropout_rate,
            merge_method,
            cgmlp_weight[lnum])
            for lnum in range(num_blocks)
        ])

        self.after_norm = nn.LayerNorm(output_size)

    def output_size(self) -> int:
        return self._output_size

    def forward(
            self,
            xs: torch.Tensor,
            ilens: torch.Tensor,
            masks: torch.Tensor,
    ) -> tuple[Any, Tensor, Optional[Any]]:
        xs, pos_emb = self.pos_enc(xs)
        mask_pad = masks  # (B, 1, T/subsample_rate)
        counts = []
        route_probs = []
        n_dropped = []
        route_prob_max = []
        for layer in self.encoders:
            xs, mask_pad, info = layer(xs, mask_pad, pos_emb, masks)
            counts.append(info[0])
            route_probs.append(info[1])
            n_dropped.append(info[2])
            route_prob_max.append(info[3])
        xs = self.after_norm(xs)
        info = (torch.stack(counts), torch.stack(route_probs), n_dropped, torch.stack(route_prob_max))
        return xs, masks, info
