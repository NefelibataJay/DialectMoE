import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union, Any

from torch import Tensor

from moe_model.experts_feedforward import ExpertsFeedForward
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

class MoeBranchformErncoderLayer(nn.Module):
    def __init__(
            self,
            size: int,
            attn: Optional[torch.nn.Module],
            cgmlp: Optional[torch.nn.Module],
            eff: Optional[torch.nn.Module],
            dropout_rate: float = 0.1,
            use_conv_fusion: bool = False,
            merge_conv_kernel: int = 31,
    ):
        super().__init__()

        self.size = size
        self.attn = attn
        self.cgmlp = cgmlp
        self.eff = eff
        self.use_conv_fusion = use_conv_fusion

        if attn is not None:
            self.norm_mha = nn.LayerNorm(size)  # for the MHA module
        if cgmlp is not None:
            self.norm_mlp = nn.LayerNorm(size)  # for the MLP module
        if eff is not None:
            self.norm_experts = nn.LayerNorm(size)  # for the experts module
        self.norm_final = nn.LayerNorm(size)  # for the final output of the block
        self.dropout = torch.nn.Dropout(dropout_rate)
        if self.use_conv_fusion:
            self.depthwise_conv_fusion = torch.nn.Conv1d(
                size + size,
                size + size,
                kernel_size=merge_conv_kernel,
                stride=1,
                padding=(merge_conv_kernel - 1) // 2,
                groups=size + size,
                bias=True,
            )
        self.merge_proj = torch.nn.Linear(size + size, size)

    def forward(
            self,
            x: torch.Tensor,
            mask: torch.Tensor,
            pos_emb: torch.Tensor,
            mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
            embed: Optional[torch.Tensor] = None,
            att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
            cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
    ) -> Tuple[Any, Tensor, tuple]:
        stoch_layer_coeff = 1.0
        # Two branches
        x1 = x
        x2 = x

        # Branch 1: multi-headed attention module
        if self.attn is not None:
            x1 = self.norm_mha(x1)
            x_att, _ = self.attn(x1, x1, x1, mask, pos_emb, att_cache)
            x1 = self.dropout(x_att)

        # Branch 2: convolutional gating mlp
        # Fake new cnn cache here, and then change it in conv_module
        if self.cgmlp is not None:
            x2 = self.norm_mlp(x2)
            x2, _ = self.cgmlp(x2, mask_pad, cnn_cache)
            x2 = self.dropout(x2)

        # merge
        x_concat = torch.cat([x1, x2], dim=-1)
        if self.use_conv_fusion:
            x_tmp = x_concat.transpose(1, 2)
            x_tmp = self.depthwise_conv_fusion(x_tmp)
            x_tmp = x_tmp.transpose(1, 2)
            x = x + self.dropout(self.merge_proj(x_concat + x_tmp))
        else:
            x = x + self.dropout(self.merge_proj(x_concat))

        if self.eff is not None:
            x = self.norm_experts(x)
            x, aux_loss = self.eff(x, embed)
            x = x + self.dropout(x)
        else:
            aux_loss = torch.tensor([0.0])
        x = self.norm_final(x)
        return x, mask, aux_loss


class MoeBranchformerEncoder(nn.Module):
    def __init__(self,
                 output_size: int = 256,
                 embed_dim: int = 256,
                 num_blocks: int = 8,
                 attention_heads: int = 4,
                 attention_layer_type: str = "rel_selfattn",
                 pos_enc_layer_type: str = "rel_pos",
                 cgmlp_linear_units: int = 1024,
                 cgmlp_conv_kernel: int = 31,
                 use_linear_after_conv: bool = False,
                 gate_activation: str = "identity",
                 num_expert: int = 4,
                 expert_size: int = 1024,
                 capacity_factor: float = 1.25,
                 drop_tokens: bool = True,
                 is_scale_prob: bool = True,
                 fusion_type: str = "none",
                 use_conv_fusion: bool = False,
                 merge_conv_kernel: int = 31,
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

        cgmlp_layer = ConvolutionalGatingMLP
        cgmlp_layer_args = (
            output_size,
            cgmlp_linear_units,
            cgmlp_conv_kernel,
            dropout_rate,
            use_linear_after_conv,
            gate_activation,
        )
        eff_layer = ExpertsFeedForward
        eff_layer_args = (output_size,
                          embed_dim,
                          expert_size,
                          num_expert,
                          dropout_rate,
                          capacity_factor,
                          drop_tokens,
                          is_scale_prob,
                          fusion_type)

        self.encoders = nn.ModuleList([MoeBranchformErncoderLayer(
            output_size,
            encoder_selfattn_layer(*encoder_selfattn_layer_args),
            cgmlp_layer(*cgmlp_layer_args),
            eff_layer(*eff_layer_args),
            dropout_rate,
            use_conv_fusion,
            merge_conv_kernel
        ) for _ in range(num_blocks)
        ])

        self.after_norm = nn.LayerNorm(output_size)

    def output_size(self) -> int:
        return self._output_size

    def forward(
            self,
            xs: torch.Tensor,
            masks: torch.Tensor,
            embed: Optional[torch.Tensor] = None,
    ) -> Tuple[Any, Tensor, Optional[Any]]:
        xs, pos_emb = self.pos_enc(xs)
        mask_pad = masks  # (B, 1, T/subsample_rate)
        aux_loss_collection = []
        for layer in self.encoders:
            xs, mask_pad, aux_loss = layer(xs, mask_pad, pos_emb, masks, embed)
            aux_loss_collection.append(aux_loss)
        xs = self.after_norm(xs)
        return xs, masks, aux_loss_collection
