from typing import Tuple

import torch


class DomainClassifier(torch.nn.Module):
    def __init__(
            self,
            accent_num: int,
            encoder_output_size: int,
            dropout_rate: float = 0.0,
    ):
        super().__init__()
        eprojs = encoder_output_size
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.fc = torch.nn.Linear(eprojs, accent_num)
        self.CEloss = torch.nn.CrossEntropyLoss(reduction="mean")
        # MSELoss = torch.nn.MSELoss(reduction="mean")

    def forward(self, encoder_output: torch.Tensor, ture_domain: torch.Tensor) -> torch.Tensor:
        """Calculate classification loss.
        Args:
            encoder_output: batch of padded hidden state sequences (B, Tmax, D)
            ture_domain: batch of accent id sequence tensor (B)
        """

        ys_hat = self.fc(self.dropout(encoder_output.sum(dim=1)))
        # Compute loss
        loss = self.CEloss(ys_hat, ture_domain)
        return loss

    def argmax(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoder_output: batch of padded hidden state sequences (B, Tmax, D)
        return:
            torch.Tensor: argmax (B, accent_num) -> (B)
        """
        return torch.argmax(self.fc(encoder_output.sum(dim=1)), dim=-1)
