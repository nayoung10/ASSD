from omegaconf import OmegaConf

from dataclasses import dataclass, field
from typing import List

import torch
from src.models import register_model
from src.models.generator import sample_from_categorical

import esm
import loralib as lora

@dataclass
class ESMAdapterConfig:
    name: str = 'esm2_t33_650M_UR50D'
    initialize_input: bool = True

@register_model('esm_adapter')
class ESMAdapter(torch.nn.Module):
    _default_cfg = ESMAdapterConfig()

    def __init__(self, cfg) -> None:
        super().__init__()
        self._update_cfg(cfg)

        self.decoder, _ = esm.pretrained.load_model_and_alphabet_hub(self.cfg.name)
        print("INFO:: Loaded ESM model version: ", self.cfg.name)

        if '650M' in self.cfg.name:
            print("INFO:: Marking only LORA as trainable")
            lora.mark_only_lora_as_trainable(self.decoder)
        else: 
            print("INFO:: Marking all parameters as trainable")
            for name, param in self.decoder.named_parameters(): 
                if not param.requires_grad:
                    param.requires_grad = True
        
        ###### Training from scratch ######
        # self.decoder._init_submodules() # initialize the submodules
        # for name, param in self.decoder.named_parameters(): 
        #     if not param.requires_grad:
        #         param.requires_grad = True
        # for name, param in self.decoder.named_parameters(): 
        #     if not param.requires_grad:
        #         raise ValueError("ERROR:: param.requires_grad is False: ", name)

        self.padding_idx = self.decoder.padding_idx
        self.mask_idx = self.decoder.mask_idx
        self.cls_idx = self.decoder.cls_idx
        self.eos_idx = self.decoder.eos_idx

    def _update_cfg(self, cfg):
        self.cfg = OmegaConf.merge(self._default_cfg, cfg)

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

        esm_logits = self.decoder(
            tokens=output_tokens,
        )['logits']

        logits = esm_logits 

        _tokens, _scores = sample_from_categorical(logits, temperature=temperature)

        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])

        history.append(output_tokens.clone())

        return dict(
            output_tokens=output_tokens,
            output_scores=output_scores,
            step=step + 1,
            max_step=max_step,
            history=history
        )

    def initialize_output_tokens(self, batch):

        initial_output_tokens = batch['prev_tokens'].clone()
        initial_output_scores = torch.zeros(
            *initial_output_tokens.size(), device=initial_output_tokens.device
        )

        return initial_output_tokens, initial_output_scores
