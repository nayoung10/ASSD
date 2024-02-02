from dataclasses import dataclass, field
from typing import List

import torch
from byprot.models import register_model
from byprot.models.fixedbb import FixedBackboneDesignEncoderDecoder
from byprot.models.fixedbb.generator import sample_from_categorical
from byprot.models.fixedbb.protein_mpnn_cmlm.protein_mpnn import (
    ProteinMPNNCMLM, ProteinMPNNConfig)

from .fixedbb.lm_design.modules.esm2_adapter import ESM2WithStructuralAdatper


@dataclass
class ESM2AdapterConfig:
    encoder: ProteinMPNNConfig = field(default=ProteinMPNNConfig())
    adapter_layer_indices: List = field(default_factory=lambda: [32, ])
    separate_loss: bool = True
    name: str = 'esm2_t33_650M_UR50D'
    dropout: float = 0.1
    # ensemble_logits: bool = False


@register_model('esm2_adapter')
class ESM2Adapter(FixedBackboneDesignEncoderDecoder):
    _default_cfg = ESM2AdapterConfig()

    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        self.decoder = ESM2WithStructuralAdatper.from_pretrained(args=self.cfg, name=self.cfg.name)

        self.padding_idx = self.decoder.padding_idx
        self.mask_idx = self.decoder.mask_idx
        self.cls_idx = self.decoder.cls_idx
        self.eos_idx = self.decoder.eos_idx

    def forward(self, batch, **kwargs):
        init_pred = batch['prev_tokens']

        esm_logits = self.decoder(
            tokens=init_pred,
        )['logits']

        return esm_logits

    def forward_decoder(self, prev_decoder_out, need_attn_weights=False):
        output_tokens = prev_decoder_out['output_tokens']
        output_scores = prev_decoder_out['output_scores']
        output_masks = prev_decoder_out['output_masks']
        step, max_step = prev_decoder_out['step'], prev_decoder_out['max_step']
        temperature = prev_decoder_out['temperature']
        history = prev_decoder_out['history']

        # output_masks = output_tokens.eq(self.mask_idx)  # & coord_mask
        # output_masks = output_tokens.ne(self.padding_idx)  # & coord_mask

        esm_out = self.decoder(
            tokens=output_tokens,
            need_head_weights=need_attn_weights
        )
        
        esm_logits = esm_out['logits']
        attentions = esm_out['attentions'] if need_attn_weights else None

        logits = esm_logits  

        _tokens, _scores = sample_from_categorical(logits, temperature=temperature)

        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])

        history.append(output_tokens.clone())

        return dict(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attentions=attentions, # [B, L, H, T, T]
            step=step + 1,
            max_step=max_step,
            history=history,
        )

    def initialize_output_tokens(self, batch):

        initial_output_tokens = batch['prev_tokens'].clone() 
        initial_output_scores = torch.zeros(
            *initial_output_tokens.size(), device=initial_output_tokens.device
        )

        return initial_output_tokens, initial_output_scores
