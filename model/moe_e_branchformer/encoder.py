import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union, Any

from torch import Tensor

from model.moduel.experts_feedforward import ExpertsFeedForward
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
from wenet.transformer.subsampling import (
    Conv2dSubsampling4,
    Conv2dSubsampling6,
    Conv2dSubsampling8,
)
from wenet.transformer.positionwise_feed_forward import PositionwiseFeedForward
from wenet.utils.common import get_activation
from wenet.utils.mask import add_optional_chunk_mask, make_pad_mask

class MoeEBranchformerLayer(nn.Module):
    def __init__(
            self,
            size: int,
            attn: Optional[torch.nn.Module],
            cgmlp: Optional[torch.nn.Module],
            experts_feed_forward: Optional[torch.nn.Module],
            feed_forward_macaron: Optional[torch.nn.Module],
            dropout_rate: float = 0.1,
            merge_conv_kernel: int = 31,
    ):
        super().__init__()

        self.size = size
        self.attn = attn
        self.cgmlp = cgmlp
        
        self.experts_feed_forward = experts_feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.ff_scale = 1.0
        if self.experts_feed_forward is not None:
            self.norm_ff = nn.LayerNorm(size)
        if self.feed_forward_macaron is not None:
            self.ff_scale = 0.5
            self.norm_ff_macaron = nn.LayerNorm(size)

        self.norm_mha = nn.LayerNorm(size)  # for the MHA module
        self.norm_mlp = nn.LayerNorm(size)  # for the MLP module
        self.norm_final = nn.LayerNorm(size)  # for the final output of the block

        self.dropout = torch.nn.Dropout(dropout_rate)

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
            att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
            cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x -> (batch, time, size)
            mask -> (batch, 1, time)
            pos_emb -> (1, time, size)
        Returns:
            torch.Tensor: Output tensor (batch, time, size)
            torch.Tensor: Mask tensor (batch, time)
        """
        # Feed forward macaron module
        if self.feed_forward_macaron is not None:
            x = x + self.ff_scale * self.dropout(self.feed_forward_macaron(self.norm_ff_macaron(x)))

        # Two branches
        x1 = x
        x2 = x

        # Branch 1: multi-headed attention module
        x1 = self.norm_mha(x1)
        x_att, _ = self.attn(x1, x1, x1, mask, pos_emb, att_cache)
        x1 = self.dropout(x_att)

        # Branch 2: convolutional gating mlp
        # Fake new cnn cache here, and then change it in conv_module
        x2 = self.norm_mlp(x2)
        x2, _ = self.cgmlp(x2, mask_pad, cnn_cache)
        x2 = self.dropout(x2)

        # Merge two branches
        x_concat = torch.cat([x1, x2], dim=-1)
        x_tmp = x_concat.transpose(1, 2)
        x_tmp = self.depthwise_conv_fusion(x_tmp)
        x_tmp = x_tmp.transpose(1, 2)
        x = x + self.dropout(self.merge_proj(x_concat + x_tmp))

        # Feed forward module
        if self.experts_feed_forward is not None:
            redisual = x
            x, aux_loss = self.experts_feed_forward(self.norm_ff(x))
            x = redisual + self.dropout(x) * self.ff_scale
        else:
            aux_loss = torch.tensor([0.0])
        x = self.norm_final(x)
        
        return x, mask,aux_loss


class MoeEBranchformerEncoder(nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int = 256,
                 num_blocks: int = 12,
                 attention_heads: int = 4,
                 attention_layer_type: str = "rel_selfattn",
                 pos_enc_layer_type: str = "rel_pos",
                 cgmlp_linear_units: int = 1024,
                 cgmlp_conv_kernel: int = 31,
                 use_linear_after_conv: bool = False,
                 gate_activation: str = "identity",
                 use_ffn: bool = False,
                 macaron_ffn: bool = False,
                 linear_units: int = 1024,
                 activation_type: str = "swish",
                 merge_conv_kernel: int = 31,
                 num_expert: int = 4,
                 expert_size: int = 1024,
                 capacity_factor: float = 1.25,
                 drop_tokens: bool = True,
                 is_scale_prob: bool = True,
                 fusion_type: str = "none",
                 input_layer : str = "conv2d",
                 dropout_rate: float = 0.1,
                 positional_dropout_rate: float = 0.1,
                 attention_dropout_rate: float = 0.0,
                 global_cmvn: torch.nn.Module = None,
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

        if input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(input_size, output_size),
                torch.nn.LayerNorm(output_size),
                torch.nn.Dropout(dropout_rate),
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling4(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer == "conv2d6":
            self.embed = Conv2dSubsampling6(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer == "conv2d8":
            self.embed = Conv2dSubsampling8(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate),
            )
        else:
            raise ValueError("unknown input_layer: " + input_layer)

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

        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (
            output_size,
            linear_units,
            dropout_rate,
            get_activation(activation_type),
        )

        experts_layer = ExpertsFeedForward
        experts_layer_args = (output_size,
                                output_size,
                                expert_size,
                                num_expert,
                                dropout_rate,
                                capacity_factor,
                                drop_tokens,
                                is_scale_prob,
                                fusion_type)

        self.encoders = nn.ModuleList([MoeEBranchformerLayer(
            output_size,
            encoder_selfattn_layer(*encoder_selfattn_layer_args),
            cgmlp_layer(*cgmlp_layer_args),
            experts_layer(*experts_layer_args) if use_ffn else None,
            positionwise_layer(*positionwise_layer_args) if use_ffn and macaron_ffn else None,
            dropout_rate,
            merge_conv_kernel,
        ) for _ in range(num_blocks)
        ])

        self.after_norm = nn.LayerNorm(output_size)
        self.global_cmvn = global_cmvn

    def output_size(self) -> int:
        return self._output_size

    def forward(
            self,
            xs: torch.Tensor,
            ilens: torch.Tensor,
    ) -> Tuple[Tensor, Tensor,]:
        T = xs.size(1)
        masks = ~make_pad_mask(ilens, T).unsqueeze(1)  # (B, 1, T)
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        xs, pos_emb, masks = self.embed(xs, masks)
        mask_pad = masks  # (B, 1, T/subsample_rate)
        chunk_masks = masks
        aux_loss_collection = []
        for layer in self.encoders:
            xs, chunk_masks,aux_loss = layer(xs, chunk_masks, pos_emb, mask_pad)
            aux_loss_collection.append(aux_loss)
        xs = self.after_norm(xs)
        return xs, masks, aux_loss_collection
