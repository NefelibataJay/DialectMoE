import torch
import yaml
from torch.utils.data import DataLoader
import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from wenet.dataset.dataset import Dataset
from moe_model.domain_encoder import DomainConformerEncoder
from moe_model.domin_classification import DomainClassifier
from moe_model.moe_branchformer_encoder import MoeBranchformerEncoder
from moe_model.me_model import MAModel
from moe_model.frontend import Frontend
from wenet.branchformer.encoder import BranchformerEncoder
from wenet.transformer.asr_model import ASRModel
from wenet.transformer.cmvn import GlobalCMVN
from wenet.transformer.ctc import CTC
from wenet.transformer.decoder import TransformerDecoder
from wenet.transformer.encoder import ConformerEncoder, TransformerEncoder
from wenet.utils.cmvn import load_cmvn
from wenet.utils.file_utils import *

def init_model(configs):
    if configs['cmvn_file'] is not None:
        mean, istd = load_cmvn(configs['cmvn_file'], configs['is_json_cmvn'])
        global_cmvn = GlobalCMVN(
            torch.from_numpy(mean).float(),
            torch.from_numpy(istd).float())
    else:
        global_cmvn = None

    input_dim = configs['input_dim']
    vocab_size = configs['output_dim']
    domain_num = configs["domain_num"]

    encoder_type = configs.get('encoder', 'conformer')
    decoder_type = configs.get('decoder', 'transformer')
    model_type = configs.get('model_type', 'wenet')

    frontend = Frontend(input_dim,
                            output_size=configs['encoder_conf']["output_size"],
                            input_layer=configs['input_layer'],
                            global_cmvn=global_cmvn)
    domain_encoder_type = configs.get('domain_encoder', 'none')
    if domain_encoder_type != 'none':
        assert configs['domain_conf']['output_size'] == configs['encoder_conf']["output_size"]
        domain_encoder = DomainConformerEncoder(**configs['domain_conf'])
        domain_classifier = DomainClassifier(accent_num=domain_num,
                                                encoder_output_size=domain_encoder.output_size())
    else:
        domain_encoder = None
        domain_classifier = None

    asr_encoder = MoeBranchformerEncoder(**configs['encoder_conf'])

    fusion = configs.get('fusion', None)
    ctc = CTC(vocab_size, asr_encoder.output_size())

    decoder = TransformerDecoder(vocab_size, asr_encoder.output_size(),
                                    **configs['decoder_conf'])

    model = MAModel(vocab_size=vocab_size,
                    frontend=frontend,
                    domain_encoder=domain_encoder,
                    asr_encoder=asr_encoder,
                    decoder=decoder,
                    domain_classifier=domain_classifier,
                    ctc=ctc,
                    fusion=fusion,
                    **configs['model_conf'])
    return model

def test_load_model():
    config_path = "conf/train_branchformer_moe.yaml"
    with open(config_path, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    data_type = "raw"
    data_root = "data/aishell/"
    train_data = os.path.join(data_root, "train/data.list")
    symbol_path = "data/dict/"
    model_dir = "exp/branchformer_expert_aishell-grad_accu_4_50000lr"
    checkpoint_path = "./exp/aishell_branchformer/avg_30.pt"
    cmvn_path = None
        
    symbol_table = read_symbol_table(symbol_path)
    domain_table = read_domain_table(symbol_path)

    vocab_size = len(symbol_table)
    domain_num = len(domain_table)
    
    if 'fbank_conf' in configs['dataset_conf']:
        input_dim = configs['dataset_conf']['fbank_conf']['num_mel_bins']
    else:
        input_dim = configs['dataset_conf']['mfcc_conf']['num_mel_bins']
    
    train_conf = configs['dataset_conf']
    pin_memory = True
    num_workers = 1
    prefetch = 500

    train_dataset = Dataset(data_type, train_data, symbol_table, domain_table, train_conf)
    
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=None,
                                   pin_memory=pin_memory,
                                   num_workers=num_workers,
                                   prefetch_factor=prefetch)
    configs['input_dim'] = input_dim
    configs['output_dim'] = vocab_size
    configs['cmvn_file'] = cmvn_path
    configs['is_json_cmvn'] = True
    configs["domain_num"] = domain_num

    # init model
    model = init_model(configs)
    
    load_param_list = load_param(model,checkpoint_path)
    
    print(model)
        
        

def load_param(model,load_path):
        param_dict = torch.load(load_path, map_location='cpu')
        model_dict = model.state_dict()
        load_param_list = []
        for k, v in model_dict.items():
            if k in param_dict and param_dict[k].size() == v.size():
                model_dict[k] = param_dict[k]
                load_param_list.append(k)
            elif "experts" in k:
                ori_k = k.replace("experts.", "")
                if ori_k in param_dict and param_dict[ori_k].size() == v.size()[1:]:
                    model_dict[k] = param_dict[ori_k].unsqueeze(0).expand(v.size())
                    load_param_list.append(k)
        load_param_list.sort()
        model.load_state_dict(model_dict)
        return load_param_list



if __name__ == "__main__":
    test_load_model()    
