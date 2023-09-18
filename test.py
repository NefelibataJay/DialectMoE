import copy

import torch
import yaml
from torch.utils.data import DataLoader

from wenet.dataset.dataset import Dataset
from wenet.utils.file_utils import *
from wenet.utils.init_model import init_model


def main():
    torch.manual_seed(777)
    mode = "attention"
    data_type = "raw"
    symbol_path = "data/dict/"
    config_path = "exp/mt_conformer_DID_ASR/train.yaml"
    test_data = "data/mandarin+kunming+sichuan/test/data.list"
    checkpoint_path = "exp/mt_conformer_DID_ASR/final.pt"
    result_file = "exp/mt_conformer_DID_ASR/test_result.txt"

    with open(config_path, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

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
    accent_table = read_accent_table(symbol_path)

    test_dataset = Dataset(data_type,
                           test_data,
                           symbol_table,
                           accent_table,
                           test_conf,
                           partition=False)

    test_data_loader = DataLoader(test_dataset, batch_size=None, num_workers=0)

    model = init_model(configs)

    char_dict = {v: k for k, v in symbol_table.items()}
    eos = len(char_dict) - 1

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint, strict=False)

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
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

            content = []
            # batch_size = 1
            for w in hyps[0]:
                if w == eos:
                    break
                content.append(char_dict[w])
            fout.write('{} {}\n'.format(keys[0], "".join(content)))

if __name__ == "__main__":
    main()
