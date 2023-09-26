import torch
from torch import nn


class Fusion(nn.Module):
    def __init__(self, size1, size2, output_size,
                 dropout_rate: float = 0.0,
                 fusion_type: str = 'concat', ):
        super().__init__()
        assert fusion_type in ["concat", "add", "attention"]
        if fusion_type == "concat":
            input_dim = size1 + size2
        elif fusion_type == "add":
            assert size1 == size2
            input_dim = size1
        elif fusion_type == "attention":
            pass
            # TODO add attention
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        self.fc = nn.Linear(input_dim, output_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.fusion_type = fusion_type

    def forward(self, input1, input2):
        if self.fusion_type == "concat":
            out = torch.cat([input1, input2], dim=-1)
        elif self.fusion_type == "add":
            out = input1 + input2
        elif self.fusion_type == "attention":
            pass
            # TODO add attention
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")
        out = self.fc(out)
        out = self.dropout(out)
        return out
