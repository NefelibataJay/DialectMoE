import torch


from typing import Optional


class DomainModel(torch.nn.Module):
    def __init__(self,domain_num: int,
                domain_encoder: torch.nn.Module,
                domain_classifier: torch.nn.Module,
                ):
        super().__init__()
        self.domain_num = domain_num
        self.domain_encoder = domain_encoder
        self.domain_classifier = domain_classifier
    
    def forward(self,
                speech: torch.Tensor,
                speech_lengths: torch.Tensor,
                domain_id: torch.Tensor):
        
        domain_encoder_output, mask = self.domain_encoder(speech, speech_lengths)
        loss_domain = self.domain_classifier(domain_encoder_output, domain_id)

        return {"loss": loss_domain}
    
    def recognize(self,
                speech: torch.Tensor,
                speech_lengths: torch.Tensor,):
        domain_encoder_output, mask = self.domain_encoder(speech, speech_lengths)
        domain_id = self.domain_classifier.argmax(domain_encoder_output)
        return domain_id