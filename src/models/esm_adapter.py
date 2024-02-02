from dataclasses import dataclass, field
from typing import List

import torch
from byprot.models import register_model
from byprot.models.fixedbb import FixedBackboneDesignEncoderDecoder
from byprot.models.fixedbb.generator import sample_from_categorical
from byprot.models.fixedbb.protein_mpnn_cmlm.protein_mpnn import (
    ProteinMPNNCMLM, ProteinMPNNConfig)

from .fixedbb.lm_design.modules.esm_adapter import ProteinBertModelWithStructuralAdatper
import esm
import loralib as lora
@dataclass
class ESMAdapterConfig:
    encoder: ProteinMPNNConfig = field(default=ProteinMPNNConfig())
    adapter_layer_indices: List = field(default_factory=lambda: [32, ])
    separate_loss: bool = True
    name: str = 'esm1b_t33_650M_UR50S'
    # ensemble_logits: bool = False
    initialize_input: bool = True


@register_model('esm_adapter')
class ESMAdapter(FixedBackboneDesignEncoderDecoder):
    _default_cfg = ESMAdapterConfig()

    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        self.decoder, _ = esm.pretrained.load_model_and_alphabet_hub(self.cfg.name)
        print("INFO:: Loaded ESM model version: ", self.cfg.name)
        ###### LoRA fine-tuning ######
        # lora.mark_only_lora_as_trainable(self.decoder)

        # ###### Full finetune ######
        print("INFO:: Finetuning the whole model")
        for name, param in self.decoder.named_parameters(): 
            if not param.requires_grad:
                param.requires_grad = True
        for name, param in self.decoder.named_parameters(): 
            if not param.requires_grad:
                raise ValueError("ERROR:: param.requires_grad is False: ", name)
        
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
