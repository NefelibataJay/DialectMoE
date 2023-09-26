import torch
import torch.nn as nn

class SparseL1Loss(nn.Module):
    def __init__(self):
        super(SparseL1Loss, self).__init__()

    def forward(self, prob_matrix: torch.Tensor, eps:float = 1e-20) -> torch.Tensor:
        prob_matrix = prob_matrix.view(-1, prob_matrix.size(-1))
        n_samples = prob_matrix.size(0)
        norm_y = prob_matrix.norm(2, dim=-1, keepdim=True)
        norm_y = torch.clamp(norm_y, min=eps)
        norm_prob = prob_matrix / norm_y
        l1_loss = norm_prob.norm(1)
        l1_loss = l1_loss / n_samples
        # l1_loss = l1_loss / n_samples
        return l1_loss


class BalanceImportanceLoss(nn.Module):
    def __init__(self):
        super(BalanceImportanceLoss, self).__init__()

    def forward(self, prob_matrix: torch.Tensor)-> torch.Tensor:
        prob_matrix = prob_matrix.view(-1, prob_matrix.size(-1))
        n_experts = prob_matrix.size(1)
        mean_prob = prob_matrix.mean(dim=0)
        importance_loss = torch.sum(mean_prob * mean_prob) * n_experts
        return importance_loss
