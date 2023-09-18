import torch

from moe_model.domain_encoder import DomainConformerEncoder
from moe_model.domin_classification import DomainClassifier
from moe_model.moe_branchformer_encoder import MoeBranchformerEncoder
from moe_model.mt_model import MAModel
from moe_model.frontend import Frontend
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

    encoder_type = configs.get('encoder', 'conformer')
    decoder_type = configs.get('decoder', 'transformer')
    model_type = configs.get('model_type', 'wenet')

    if model_type == "wenet":
        if encoder_type == 'conformer':
            encoder = ConformerEncoder(input_dim,
                                       global_cmvn=global_cmvn,
                                       **configs['encoder_conf'])
        elif encoder_type == 'branchformer':
            encoder = BranchformerEncoder(input_dim,
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
            raise ValueError("unknown decoder_type: " + decoder_type)

        model = ASRModel(vocab_size=vocab_size,
                         encoder=encoder,
                         decoder=decoder,
                         ctc=ctc,
                         lfmmi_dir=configs.get('lfmmi_dir', ''),
                         **configs['model_conf'])

    elif model_type == "moe_model":
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
    else:
        raise ValueError("unknown model_type: " + model_type)
    return model
