from typing import Tuple

import torch
import torch.nn.functional as F
import torch.nn as nn


class Accentclassification(torch.nn.Module):
    def __init__(
            self,
            accent_num: int,
            encoder_output_size: int,
            dropout_rate: float = 0.0,
    ):
        super().__init__()
        eprojs = encoder_output_size
        self.dropout_rate = dropout_rate
        self.fc = torch.nn.Linear(eprojs, accent_num)
        self.CEloss = torch.nn.CrossEntropyLoss(reduction="mean")

    def forward(self, encoder_output: torch.Tensor, ture_accent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate classification loss.
        Args:
            encoder_output: batch of padded hidden state sequences (B, Tmax, D)
            ture_accent: batch of accent id sequence tensor (B)
        """
        encoder_output = F.dropout(encoder_output, p=self.dropout_rate)
        ys_hat = encoder_output.sum(dim=1)
        ys_hat = self.fc(ys_hat)
        # Compute loss
        loss = self.CEloss(torch.log_softmax(ys_hat, dim=1), ture_accent)
        return loss, ys_hat

    def argmax(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoder_output: batch of padded hidden state sequences (B, Tmax, D)
        return:
            torch.Tensor: argmax (B, accent_num) -> (B)
        """
        return torch.argmax(self.fc(encoder_output.sum(dim=1)), dim=-1)
