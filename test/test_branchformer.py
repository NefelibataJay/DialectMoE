import torch

from moe_model.domain_encoder import DomainConformerEncoder
from moe_model.moe_branchformer_encoder import MoeBranchformerEncoder
from wenet.utils.mask import make_pad_mask


def test_branchformer():
    input_dim = 80
    num_blocks = 4
    domain_encoder = DomainConformerEncoder(input_dim)
    encoder = MoeBranchformerEncoder(input_dim, embed_dim=input_dim,fusion_type="concat",num_blocks=num_blocks)
    print(encoder)
    x = torch.randn(2, 3, input_dim)
    x_len = torch.tensor([3, 3])
    masks = ~make_pad_mask(x_len, x.size(1)).unsqueeze(1)
    embed, masks = domain_encoder(x, x_len, masks)
    xs, masks, aux_loss = encoder(x, x_len, masks, embed)



    aux_scale = [0.15, 0.15]
    loss = 0.0
    for aux in aux_loss:
        l1_loss, importance_loss = aux
        loss += aux_scale[0] * l1_loss
        loss += aux_scale[1] * importance_loss




