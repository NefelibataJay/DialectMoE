import torch
import yaml

from init_model import init_model


def test_model():
    config_path = "../conf/train_branchformer_moe.yaml"
    with open(config_path, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    configs["domain_num"] = 2
    configs['input_dim'] = 80
    configs['output_dim'] = 10
    configs['cmvn_file'] = None
    model = init_model(configs)

    x = torch.randn(2, 35, 80)
    x_len = torch.tensor([35, 35])
    target = torch.tensor([[3, 2, 6, -1], [3, 5, 7, 8]])
    target_len = torch.tensor([3, 4])
    domain_label = torch.tensor([0, 1])

    info = model(x, x_len, target, target_len, domain_label)

    print(info)
