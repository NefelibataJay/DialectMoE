import torch
from model.asr_model import MyASRModel
from model.domain_model import DomainModel
from model.ebranchformer.encoder import EBranchformerEncoder
from model.moduel.domin_classification import DomainClassifier
from model.moduel.fusion import Fusion
from model.moe_asr_model import MoeAsrModel
from model.moe_conformer.encoder import MoeConformerEncoder
from model.moe_e_branchformer.encoder import MoeEBranchformerEncoder

from wenet.branchformer.encoder import BranchformerEncoder
from wenet.transformer.asr_model import ASRModel
from wenet.transformer.cmvn import GlobalCMVN
from wenet.transformer.ctc import CTC
from wenet.transformer.decoder import TransformerDecoder
from wenet.transformer.encoder import ConformerEncoder, TransformerEncoder
from wenet.utils.cmvn import load_cmvn


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

    domain_encoder_type = configs.get('domain_encoder', 'none')
    encoder_type = configs.get('encoder', 'conformer')
    decoder_type = configs.get('decoder', 'none')
    model_type = configs.get('model_type', 'my_model')

    if model_type == "moe_model":
        if domain_encoder_type == "conformer":
            domain_encoder = ConformerEncoder(input_dim,
                                       global_cmvn=None,
                                       **configs['domain_encoder_conf'])
            domain_classifier = DomainClassifier(domain_num, domain_encoder.output_size())
        else:
            domain_encoder = None
            domain_classifier=None
        
        if encoder_type == 'conformer':
            encoder = MoeConformerEncoder(input_dim,
                                       global_cmvn=global_cmvn,
                                       **configs['encoder_conf'])
        elif encoder_type == "ebranchformer":
            encoder = MoeEBranchformerEncoder(input_size=input_dim,
                                           global_cmvn=global_cmvn,
                                           **configs['encoder_conf'])
        else:
            raise ValueError(f"encoder type was not support {encoder_type}")
            
        ctc = CTC(vocab_size, encoder.output_size())
        
        if decoder_type == 'transformer':
            decoder = TransformerDecoder(vocab_size, encoder.output_size(),
                                         **configs['decoder_conf'])
        else:
            decoder = None

        # fusion = Fusion(domain_encoder.output_size(),encoder.output_size(),encoder.output_size())
        fusion =None

        model = MoeAsrModel(vocab_size,
                            encoder=encoder,
                            decoder=decoder,
                            ctc=ctc,
                            domain_encoder=domain_encoder,
                            domain_classifier=domain_classifier,
                            fusion=fusion,
                            **configs["model_conf"])
        
    elif model_type == "my_model":
        if encoder_type == 'conformer':
            encoder = ConformerEncoder(input_dim,
                                       global_cmvn=global_cmvn,
                                       **configs['encoder_conf'])
        elif encoder_type == 'branchformer':
            encoder = BranchformerEncoder(input_dim,
                                          global_cmvn=global_cmvn,
                                          **configs['encoder_conf'])
        elif encoder_type == "ebranchformer":
            encoder = EBranchformerEncoder(input_size=input_dim,
                                           global_cmvn=global_cmvn,
                                           **configs['encoder_conf'])
        else:
            encoder = TransformerEncoder(input_dim,
                                         global_cmvn=global_cmvn,
                                         **configs['encoder_conf'])
        ctc = CTC(vocab_size, encoder.output_size())

        if decoder_type == 'transformer':
            decoder = TransformerDecoder(vocab_size, encoder.output_size(),
                                         **configs['decoder_conf'])
        else:
            decoder = None
        model = MyASRModel(vocab_size=vocab_size,
                         encoder=encoder,
                         decoder=decoder,
                         ctc=ctc,
                         **configs['model_conf'])
    elif model_type == "domain_asr_model":
        pass
    elif model_type == "domain_model":
        domain_encoder = ConformerEncoder(input_dim,
                                       global_cmvn=global_cmvn,
                                       **configs['domain_encoder_conf'])
        domain_classifier = DomainClassifier(domain_num, domain_encoder.output_size())

        model = DomainModel(domain_num=domain_num,
                            domain_encoder=domain_encoder,
                            domain_classifier=domain_classifier)
    else:
        raise ValueError(f"model type was not support {model_type}")
    return model
