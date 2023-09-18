from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

import torch

from wenet.transformer.ctc import CTC
from wenet.transformer.label_smoothing_loss import LabelSmoothingLoss
from wenet.utils.common import add_sos_eos, reverse_pad_list, th_accuracy, remove_duplicates_and_blank
from wenet.utils.mask import make_pad_mask, mask_finished_preds, subsequent_mask, mask_finished_scores


class MyASRModel(torch.nn.Module):
    def __init__(
            self,
            vocab_size: int,
            frontend: torch.nn.Module,
            encoder: torch.nn.Module,
            decoder: torch.nn.Module,
            ctc: CTC,
            ctc_weight: float = 0.3,
            att_weight: float = 0.7,
            ignore_id: int = -1,
            lsm_weight: float = 0.0,
            length_normalized_loss: bool = False,
    ):
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.att_weight = att_weight

        self.encoder = encoder
        self.decoder = decoder
        self.frontend = frontend
        self.ctc = ctc
        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

    def forward(
            self,
            speech: torch.Tensor,
            speech_lengths: torch.Tensor,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
    ) -> Dict[str, Optional[torch.Tensor]]:
        encoder_input, mask = self.frontend(speech, speech_lengths)
        # Encoder
        encoder_out, encoder_mask, info = self.encoder(encoder_input, speech_lengths, mask)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)

        if self.att_weight != 0.0:
            loss_att, acc_att = self._calc_att_loss(encoder_out, encoder_mask,
                                                    text, text_lengths)
        else:
            loss_att = torch.tensor(0.0)

        if self.ctc_weight != 0.0:
            loss_ctc = self.ctc(encoder_out, encoder_out_lens, text,
                                text_lengths)
        else:
            loss_ctc = torch.tensor(0.0)

        if self.moe_weight != 0.0:
            balancing_loss = self._calc_moe_loss(info)
        else:
            balancing_loss = torch.tensor(0.0)

        loss = self.ctc_weight * loss_ctc + self.att_weight * loss_att

        return {"loss": loss, "loss_att": loss_att, "loss_ctc": loss_ctc}

    def _calc_moe_loss(self, info: Tuple) -> torch.Tensor:
        counts = info[0]
        route_probs = info[1]
        total = counts.sum(dim=-1, keepdims=True)
        route_frac = counts / total
        route_prob = route_probs / total
        balancing_loss = self.encoder.num_expert * (route_frac * route_prob).sum()

        return balancing_loss

    def _calc_att_loss(
            self,
            encoder_out: torch.Tensor,
            encoder_mask: torch.Tensor,
            ys_pad: torch.Tensor,
            ys_pad_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos,
                                            self.ignore_id)
        ys_in_lens = ys_pad_lens + 1
        # 1. Forward decoder
        decoder_out, _, _ = self.decoder(encoder_out, encoder_mask,
                                         ys_in_pad, ys_in_lens,
                                         )
        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )
        return loss_att, acc_att

    def _forward_encoder(
            self,
            speech: torch.Tensor,
            speech_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Let's assume B = batch_size
        encoder_input, mask = self.frontend(speech, speech_lengths)
        # 1. Encoder
        encoder_out, encoder_mask, _ = self.encoder(
            encoder_input,
            speech_lengths,
            mask,
        )  # (B, maxlen, encoder_dim)

        return encoder_out, encoder_mask

    def recognize(
            self,
            speech: torch.Tensor,
            speech_lengths: torch.Tensor,
            beam_size: int = 10,
    ) -> tuple[Any, Any]:
        """ Apply beam search on attention decoder

        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_lengths (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
        Returns:
            torch.Tensor: decoding result, (batch, max_result_len)
        """
        assert speech.shape[0] == speech_lengths.shape[0]
        device = speech.device
        batch_size = speech.shape[0]

        # Let's assume B = batch_size and N = beam_size
        # 1. Encoder
        encoder_out, encoder_mask = self._forward_encoder(
            speech, speech_lengths,
        )  # (B, maxlen, encoder_dim)

        maxlen = encoder_out.size(1)
        encoder_dim = encoder_out.size(2)
        running_size = batch_size * beam_size
        encoder_out = encoder_out.unsqueeze(1).repeat(1, beam_size, 1, 1).view(
            running_size, maxlen, encoder_dim)  # (B*N, maxlen, encoder_dim)
        encoder_mask = encoder_mask.unsqueeze(1).repeat(
            1, beam_size, 1, 1).view(running_size, 1,
                                     maxlen)  # (B*N, 1, max_len)

        hyps = torch.ones([running_size, 1], dtype=torch.long,
                          device=device).fill_(self.sos)  # (B*N, 1)
        scores = torch.tensor([0.0] + [-float('inf')] * (beam_size - 1),
                              dtype=torch.float)
        scores = scores.to(device).repeat([batch_size]).unsqueeze(1).to(
            device)  # (B*N, 1)
        end_flag = torch.zeros_like(scores, dtype=torch.bool, device=device)
        cache: Optional[List[torch.Tensor]] = None
        # 2. Decoder forward step by step
        for i in range(1, maxlen + 1):
            # Stop if all batch and all beam produce eos
            if end_flag.sum() == running_size:
                break
            # 2.1 Forward decoder step
            hyps_mask = subsequent_mask(i).unsqueeze(0).repeat(
                running_size, 1, 1).to(device)  # (B*N, i, i)
            # logp: (B*N, vocab)
            logp, cache = self.decoder.forward_one_step(
                encoder_out, encoder_mask, hyps, hyps_mask, cache)
            # 2.2 First beam prune: select topk best prob at current time
            top_k_logp, top_k_index = logp.topk(beam_size)  # (B*N, N)
            top_k_logp = mask_finished_scores(top_k_logp, end_flag)
            top_k_index = mask_finished_preds(top_k_index, end_flag, self.eos)
            # 2.3 Second beam prune: select topk score with history
            scores = scores + top_k_logp  # (B*N, N), broadcast add
            scores = scores.view(batch_size, beam_size * beam_size)  # (B, N*N)
            scores, offset_k_index = scores.topk(k=beam_size)  # (B, N)
            # Update cache to be consistent with new topk scores / hyps
            cache_index = (offset_k_index // beam_size).view(-1)  # (B*N)
            base_cache_index = (torch.arange(batch_size, device=device).view(
                -1, 1).repeat([1, beam_size]) * beam_size).view(-1)  # (B*N)
            cache_index = base_cache_index + cache_index
            cache = [
                torch.index_select(c, dim=0, index=cache_index) for c in cache
            ]
            scores = scores.view(-1, 1)  # (B*N, 1)
            # 2.4. Compute base index in top_k_index,
            # regard top_k_index as (B*N*N),regard offset_k_index as (B*N),
            # then find offset_k_index in top_k_index
            base_k_index = torch.arange(batch_size, device=device).view(
                -1, 1).repeat([1, beam_size])  # (B, N)
            base_k_index = base_k_index * beam_size * beam_size
            best_k_index = base_k_index.view(-1) + offset_k_index.view(
                -1)  # (B*N)

            # 2.5 Update best hyps
            best_k_pred = torch.index_select(top_k_index.view(-1),
                                             dim=-1,
                                             index=best_k_index)  # (B*N)
            best_hyps_index = best_k_index // beam_size
            last_best_k_hyps = torch.index_select(
                hyps, dim=0, index=best_hyps_index)  # (B*N, i)
            hyps = torch.cat((last_best_k_hyps, best_k_pred.view(-1, 1)),
                             dim=1)  # (B*N, i+1)

            # 2.6 Update end flag
            end_flag = torch.eq(hyps[:, -1], self.eos).view(-1, 1)

        # 3. Select best of best
        scores = scores.view(batch_size, beam_size)
        # TODO: length normalization
        best_scores, best_index = scores.max(dim=-1)
        best_hyps_index = best_index + torch.arange(
            batch_size, dtype=torch.long, device=device) * beam_size
        best_hyps = torch.index_select(hyps, dim=0, index=best_hyps_index)
        best_hyps = best_hyps[:, 1:]
        return best_hyps, best_scores

    def ctc_greedy_search(
            self,
            speech: torch.Tensor,
            speech_lengths: torch.Tensor,
    ) -> tuple[Any, Any]:
        """ Apply CTC greedy search

        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_lengths (torch.Tensor): (batch, )
        Returns:
            List[List[int]]: best path result
        """
        assert speech.shape[0] == speech_lengths.shape[0]
        batch_size = speech.shape[0]
        # Let's assume B = batch_size
        encoder_out, encoder_mask = self._forward_encoder(
            speech, speech_lengths, )  # (B, maxlen, encoder_dim)
        maxlen = encoder_out.size(1)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        ctc_probs = self.ctc.log_softmax(
            encoder_out)  # (B, maxlen, vocab_size)
        topk_prob, topk_index = ctc_probs.topk(1, dim=2)  # (B, maxlen, 1)
        topk_index = topk_index.view(batch_size, maxlen)  # (B, maxlen)
        mask = make_pad_mask(encoder_out_lens, maxlen)  # (B, maxlen)
        topk_index = topk_index.masked_fill_(mask, self.eos)  # (B, maxlen)
        hyps = [hyp.tolist() for hyp in topk_index]
        scores = topk_prob.max(1)
        hyps = [remove_duplicates_and_blank(hyp) for hyp in hyps]
        return hyps, scores
