import copy
import os
import sys
import torch
import yaml
from torch.utils.data import DataLoader
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from wenet.dataset.dataset import Dataset
from wenet.utils.file_utils import *
from init_model import init_model


def main():
    torch.manual_seed(777)
    mode = "attention" # attention, ctc_greedy_search, accent_recognition
    data_type = "raw"
    symbol_path = "data/dict/"
    test_data = "data/aishell/test/data.list"
    model_dir = "exp/conformer_moe_e8"

    config_path = os.path.join(model_dir,"train.yaml")

    checkpoint_path = os.path.join(model_dir,"final.pt")

    if not os.path.exists(os.path.join(model_dir, mode)):
        os.makedirs(os.path.join(model_dir, mode))

    result_file = os.path.join(model_dir, mode, "text")
    cmvn_path =os.path.join(model_dir,"global_cmvn") if os.path.exists(os.path.join(model_dir,"global_cmvn")) else None

    with open(config_path, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    if 'fbank_conf' in configs['dataset_conf']:
        input_dim = configs['dataset_conf']['fbank_conf']['num_mel_bins']
    else:
        input_dim = configs['dataset_conf']['mfcc_conf']['num_mel_bins']

    test_conf = copy.deepcopy(configs['dataset_conf'])
    test_conf['filter_conf']['max_length'] = 102400
    test_conf['filter_conf']['min_length'] = 0
    test_conf['filter_conf']['token_max_length'] = 102400
    test_conf['filter_conf']['token_min_length'] = 0
    test_conf['filter_conf']['max_output_input_ratio'] = 102400
    test_conf['filter_conf']['min_output_input_ratio'] = 0
    test_conf['speed_perturb'] = False
    test_conf['spec_aug'] = False
    test_conf['spec_sub'] = False
    test_conf['spec_trim'] = False
    test_conf['shuffle'] = False
    test_conf['sort'] = False
    if 'fbank_conf' in test_conf:
        test_conf['fbank_conf']['dither'] = 0.0
    elif 'mfcc_conf' in test_conf:
        test_conf['mfcc_conf']['dither'] = 0.0
    test_conf['batch_conf']['batch_type'] = "static"
    test_conf['batch_conf']['batch_size'] = 1  # only support batch_size=1
    configs['cmvn_file'] = None

    symbol_table = read_symbol_table(symbol_path)
    domain_table = read_domain_table(symbol_path)

    vocab_size = len(symbol_table)
    domain_num = len(domain_table)

    test_dataset = Dataset(data_type,
                           test_data,
                           symbol_table,
                           domain_table,
                           test_conf,
                           partition=False)

    test_data_loader = DataLoader(test_dataset, batch_size=None, num_workers=0)
    configs['input_dim'] = input_dim
    configs['output_dim'] = vocab_size
    configs['cmvn_file'] = cmvn_path
    configs['is_json_cmvn'] = True
    configs["domain_num"] = domain_num

    model = init_model(configs)

    char_dict = {v: k for k, v in symbol_table.items()}
    eos = len(char_dict) - 1

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint, strict=False)

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    # device = torch.device('cpu')
    model = model.to(device)

    model.eval()

    with torch.no_grad(), open(result_file, 'w') as fout:
        for batch_idx, batch in enumerate(test_data_loader):
            keys, feats, target, feats_lengths, target_lengths, accent_id = batch
            feats = feats.to(device)
            target = target.to(device)
            feats_lengths = feats_lengths.to(device)
            target_lengths = target_lengths.to(device)
            accent_id = accent_id.to(device)
            if mode == 'attention':
                hyps, _ = model.recognize(
                    feats,
                    feats_lengths,
                    beam_size=10,
                )
                hyps = [hyp.tolist() for hyp in hyps]
            elif mode == 'ctc_greedy_search':
                hyps, _ = model.ctc_greedy_search(
                    feats,
                    feats_lengths,
                )
            elif model == 'accent_recognition':
                pass
            for i, key in enumerate(keys):
                content = []
                for w in hyps[i]:
                    if w == eos:
                        break
                    content.append(char_dict[w])
                print('{} {}'.format(key, "".join(content)))
                fout.write('{} {}\n'.format(key,"".join(content)))
            

if __name__ == "__main__":
    main()
