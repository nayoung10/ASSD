import os
from typing import Any, Callable, List, Union
from pathlib import Path
import numpy as np
import torch
from src import utils
from src.models.generator import IterativeRefinementGenerator, sample_from_categorical
from src.modules import metrics
from src.tasks import TaskLitModule, register_task
from src.utils.config import compose_config as Cfg, merge_config

from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F
from torchmetrics import CatMetric, MaxMetric, MeanMetric, MinMetric

from src.datamodules.datasets.data_utils import Alphabet

# import esm

log = utils.get_logger(__name__)


def new_arange(x, *size):
    """
    Return a Tensor of `size` filled with a range function on the device of x.
    If size is empty, using the size of the variable x.
    """
    if len(size) == 0:
        size = x.size()
    return torch.arange(size[-1], device=x.device).expand(*size).contiguous()


@register_task('model_cdr')
class CMLM(TaskLitModule):

    _DEFAULT_CFG: DictConfig = Cfg(
        learning=Cfg(
            noise='no_noise',  # ['full_mask', 'random_mask']
            num_unroll=0,
        ),
        generator=Cfg(
            max_iter=1,
            strategy='denoise',  # ['denoise' | 'mask_predict']
            noise='full_mask_cdr',  # ['full_mask' | 'selected mask']
            replace_visible_tokens=False,
            temperature=0,
            eval_sc=False,
        )
    )

    def __init__(
        self,
        model: Union[nn.Module, DictConfig],
        alphabet: DictConfig,
        criterion: Union[nn.Module, DictConfig],
        optimizer: DictConfig,
        lr_scheduler: DictConfig = None,
        *,
        learning=_DEFAULT_CFG.learning,
        generator=_DEFAULT_CFG.generator
    ):
        super().__init__(model, criterion, optimizer, lr_scheduler)

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        # self.save_hyperparameters(ignore=['model', 'criterion'], logger=False)
        self.save_hyperparameters(logger=True)

        self.alphabet = Alphabet(**alphabet)
        self.build_model() 
        self.build_generator()

    def setup(self, stage=None) -> None:
        super().setup(stage)

        self.build_criterion()
        self.build_torchmetric()

        if self.stage == 'fit':
            log.info(f'\n{self.model}')

    def build_model(self):
        log.info(f"Instantiating neural model <{self.hparams.model._target_}>")
        self.model = utils.instantiate_from_config(cfg=self.hparams.model, group='model')

    def build_generator(self):
        self.hparams.generator = merge_config(
            default_cfg=self._DEFAULT_CFG.generator,
            override_cfg=self.hparams.generator
        )
        self.generator = IterativeRefinementGenerator(
            alphabet=self.alphabet,
            **self.hparams.generator
        )
        log.info(f"Generator config: {self.hparams.generator}")

    def build_criterion(self):
        self.criterion = utils.instantiate_from_config(cfg=self.hparams.criterion) 
        self.criterion.ignore_index = self.alphabet.padding_idx

    def build_torchmetric(self):
        self.eval_loss = MeanMetric()
        self.eval_nll_loss = MeanMetric()

        self.val_ppl_best = MinMetric()

        self.acc = MeanMetric()
        self.acc_best = MaxMetric()

        self.acc_median = CatMetric()
        self.acc_median_best = MaxMetric()

        self.acc_mean = MeanMetric()
        self.acc_mean_best = MaxMetric()

        # composition metrics
        self.tvd_mean = MeanMetric()
        self.cosim_mean = MeanMetric()
        self.cosim_mean_best = MaxMetric()

        # standard deviations
        self.recovery_list = []
        self.tvd_list = [] 
        self.cosim_list = []

    def load_from_ckpt(self, ckpt_path):
        state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print(f"Restored from {ckpt_path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def on_epoch_start(self) -> None:
        if self.hparams.generator.eval_sc:
            import esm
            log.info(f"Eval structural self-consistency enabled. Loading ESMFold model...")
            self._folding_model = esm.pretrained.esmfold_v1().eval()
            self._folding_model = self._folding_model.to(self.device)

    # -------# Training #-------- #
    @torch.no_grad()
    def inject_noise(self, tokens, coord_mask, cdrs=None, noise=None, sel_mask=None, mask_by_unk=False):
        padding_idx = self.alphabet.padding_idx
        if mask_by_unk:
            mask_idx = self.alphabet.unk_idx
        else:
            mask_idx = self.alphabet.mask_idx

        def _full_mask(target_tokens):
            target_mask = (
                target_tokens.ne(padding_idx)  # & mask
                & target_tokens.ne(self.alphabet.cls_idx)
                & target_tokens.ne(self.alphabet.eos_idx)
            )
            # masked_target_tokens = target_tokens.masked_fill(~target_mask, mask_idx)
            masked_target_tokens = target_tokens.masked_fill(target_mask, mask_idx)
            return masked_target_tokens

        def _random_mask(target_tokens):
            target_masks = (
                target_tokens.ne(padding_idx) & coord_mask
            )
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(~target_masks, 2.0)
            target_length = target_masks.sum(1).float()
            target_length = target_length * target_length.clone().uniform_()
            target_length = target_length + 1  # make sure to mask at least one token.

            _, target_rank = target_score.sort(1)
            target_cutoff = new_arange(target_rank) < target_length[:, None].long()
            masked_target_tokens = target_tokens.masked_fill(
                target_cutoff.scatter(1, target_rank, target_cutoff), mask_idx
            )
            return masked_target_tokens 

        def _selected_mask(target_tokens, sel_mask):
            masked_target_tokens = torch.masked_fill(target_tokens, mask=sel_mask, value=mask_idx)
            return masked_target_tokens

        def _adaptive_mask(target_tokens):
            raise NotImplementedError
        
        def _random_mask_cdr(target_tokens, cdrs, cdr_type: int):
            cdr_mask = (cdrs == cdr_type)

            # Assign random scores to cdr positions
            cdr_scores = target_tokens.clone().float().uniform_()
            cdr_scores.masked_fill_(~cdr_mask, 2.0)
            
            # Calculate length of cdr regions to be masked
            cdr_lengths = cdr_mask.sum(dim=1).float()
            mask_lengths = (cdr_lengths * cdr_lengths.clone().uniform_() + 1).long()

            _, cdr_rank = cdr_scores.sort(1)
            mask_cutoff = (torch.arange(target_tokens.size(1), device=target_tokens.device).unsqueeze(0) < mask_lengths.unsqueeze(1))
            
            # Mask tokens in cdr regions based on mask cutoff
            masked_tokens = target_tokens.masked_fill(mask_cutoff.scatter(1, cdr_rank, mask_cutoff) & cdr_mask, mask_idx)

            return masked_tokens
        
        def _full_mask_cdr(tokens, cdrs, cdr_type: int):
            cdr_mask = (cdrs == cdr_type)

            # Apply the mask to the target tokens.
            masked_target_tokens = tokens.masked_fill(cdr_mask, mask_idx)
            
            return masked_target_tokens
        
        noise = noise or self.hparams.noise

        if noise == 'full_mask':
            masked_tokens = _full_mask(tokens)
        elif noise == 'random_mask':
            masked_tokens = _random_mask(tokens)
        elif noise == 'selected_mask':
            masked_tokens = _selected_mask(tokens, sel_mask=sel_mask)
        elif noise == 'no_noise':
            masked_tokens = tokens
        elif noise == 'random_mask_cdr':
            assert cdrs is not None, "cdrs tensor must be provided for random_mask_cdr"
            masked_tokens = _random_mask_cdr(tokens, cdrs, self.hparams.learning.cdr_type)
        elif noise == 'full_mask_cdr':
            assert cdrs is not None, "cdrs tensor must be provided for full_mask_cdr"
            masked_tokens = _full_mask_cdr(tokens, cdrs, self.hparams.learning.cdr_type)
        else:
            raise ValueError(f"Noise type ({noise}) not defined.")

        prev_tokens = masked_tokens
        prev_token_mask = prev_tokens.eq(mask_idx) 
        # target_mask = prev_token_mask & coord_mask

        return prev_tokens, prev_token_mask  # , target_mask

    def step(self, batch, RL=False):
        """
        batch is a Dict containing:
            - corrds: FloatTensor [bsz, len, n_atoms, 3], coordinates of proteins
            - corrd_mask: BooltTensor [bsz, len], where valid coordinates
                are set True, otherwise False
            - lengths: int [bsz, len], protein sequence lengths
            - tokens: LongTensor [bsz, len], sequence of amino acids     
        """
        coords = batch['coords']
        coord_mask = batch['coord_mask']
        tokens = batch['tokens']
        cdrs = batch['cdrs']
        
        prev_tokens, prev_token_mask = self.inject_noise(
            tokens, coord_mask, cdrs, noise=self.hparams.learning.noise)
        batch['prev_tokens'] = prev_tokens
        batch['prev_token_mask'] = label_mask = prev_token_mask

        logits = self.model(batch)
       
        if RL:
            output_tokens, output_scores = self.rollout(logits, batch)
            log_likelihood = output_scores.sum(dim=-1)
            target_cdr_composition = metrics.get_cdr_composition(tokens, prev_token_mask, self.alphabet)
            pred_cdr_composition = metrics.get_cdr_composition(output_tokens, prev_token_mask, self.alphabet)

            # tvd_reward = -metrics.total_variation_distance(target_cdr_composition, pred_cdr_composition)
            
            # reinforce_loss = -((tvd_reward - tvd_reward.mean()) * log_likelihood).mean()
            cosim_reward = metrics.cosine_similarity(target_cdr_composition, pred_cdr_composition)
            reinforce_loss = -((cosim_reward - cosim_reward.mean()) * log_likelihood).mean()
        else:
            reinforce_loss = 0
            #cosim_per_sample = metrics.cosine_similarity(target_cdr_composition, pred_cdr_composition)
            
        loss, logging_output = self.criterion(logits, tokens, label_mask=label_mask)
        loss = loss + self.hparams.learning.alpha*reinforce_loss
    
        return loss, logging_output

    def training_step(self, batch: Any, batch_idx: int):
        loss, logging_output = self.step(batch, RL=self.hparams.learning.RL)
        
        # log train metrics
        self.log('global_step', self.global_step, on_step=True, on_epoch=False, prog_bar=True)
        self.log('lr', self.lrate, on_step=True, on_epoch=False, prog_bar=True)

        for log_key in logging_output:
            log_value = logging_output[log_key]
            self.log(f"train/{log_key}", log_value, on_step=True, on_epoch=False, prog_bar=True)

        return {"loss": loss}

    def rollout(self, guiding_logits, batch):

        # initialize output tokens and scores    
        output_tokens = batch['prev_tokens'].clone()
        output_scores = torch.zeros(
            *output_tokens.size(), device=output_tokens.device
        )
        output_masks = batch['prev_token_mask'].clone()

        _tokens, _scores = sample_from_categorical(guiding_logits, temperature=self.hparams.generator.temperature)
        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])

        return output_tokens, output_scores

    # -------# Evaluating #-------- #
    def on_test_epoch_start(self) -> None:
        self.hparams.noise = 'full_mask_cdr'

    def validation_step(self, batch: Any, batch_idx: int):
        loss, logging_output = self.step(batch, RL=self.hparams.learning.RL)

        # log other metrics
        sample_size = logging_output['sample_size']
        self.eval_loss.update(loss, weight=sample_size)
        self.eval_nll_loss.update(logging_output['nll_loss'], weight=sample_size)

        if self.stage == 'fit':
            pred_outs = self.predict_step(batch, batch_idx)
        return {"loss": loss}

    def validation_epoch_end(self, outputs: List[Any]):
        log_key = 'test' if self.stage == 'test' else 'val'

        # compute metrics averaged over the whole dataset
        eval_loss = self.eval_loss.compute()
        self.eval_loss.reset()
        eval_nll_loss = self.eval_nll_loss.compute()
        self.eval_nll_loss.reset()
        eval_ppl = torch.exp(eval_nll_loss)

        self.log(f"{log_key}/loss", eval_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{log_key}/nll_loss", eval_nll_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{log_key}/ppl", eval_ppl, on_step=False, on_epoch=True, prog_bar=True)

        if self.stage == 'fit':
            self.val_ppl_best.update(eval_ppl)
            self.log("val/ppl_best", self.val_ppl_best.compute(), on_epoch=True, prog_bar=True)

            self.predict_epoch_end(results=None)

        super().validation_epoch_end(outputs)

    # -------# Inference/Prediction #-------- #
    def forward(self, batch, return_ids=False):
        # In testing, remove target tokens to ensure no data leakage!
        # or you can just use the following one if you really know what you are doing:
        #   tokens = batch['tokens']
        tokens = batch.pop('tokens')

        prev_tokens, prev_token_mask = self.inject_noise(
            tokens, batch['coord_mask'],
            cdrs=batch['cdrs'],
            noise=self.hparams.generator.noise,  # NOTE: 'full_mask_cdr' by default. 
        )
        batch['prev_tokens'] = prev_tokens
        batch['prev_token_mask'] = prev_tokens.eq(self.alphabet.mask_idx)

        output_tokens, output_scores = self.generator.generate(
            model=self.model, batch=batch,
            max_iter=self.hparams.generator.max_iter,
            strategy=self.hparams.generator.strategy,
            replace_visible_tokens=self.hparams.generator.replace_visible_tokens,
            cdr_type = self.hparams.learning.cdr_type,
            temperature=self.hparams.generator.temperature
        )
        if not return_ids:
            return self.alphabet.decode(output_tokens)
        return output_tokens

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0, log_metrics=True) -> Any:
        # coord_mask = batch['coord_mask']
        tokens = batch['tokens']
        cdr_mask = batch['cdrs'] == self.hparams.learning.cdr_type
        
        pred_tokens = self.forward(batch, return_ids=True)

        # NOTE: use esm-1b to refine
        # pred_tokens = self.esm_refine(
        #     pred_ids=torch.where(coord_mask, pred_tokens, prev_tokens))
        # # decode(pred_tokens[0:1], self.alphabet)

        if log_metrics:
            # per-sample accuracy 
            recovery_acc_per_sample = metrics.accuracy_per_sample(pred_tokens, tokens, mask=cdr_mask)
            self.acc_median.update(recovery_acc_per_sample)
            self.acc_mean.update(recovery_acc_per_sample)

            # # global accuracy
            recovery_acc = metrics.accuracy(pred_tokens, tokens, mask=cdr_mask)
            self.acc.update(recovery_acc, weight=cdr_mask.sum())

            # Compute and log composition metrics only when not in 'fit' stage
            # if not self.stage == 'fit':
            # composition metrics
            target_cdr_composition = metrics.get_cdr_composition(tokens, cdr_mask, self.alphabet)
            pred_cdr_composition = metrics.get_cdr_composition(pred_tokens, cdr_mask, self.alphabet)

            tvd_per_sample = metrics.total_variation_distance(target_cdr_composition, pred_cdr_composition)
            self.tvd_mean.update(tvd_per_sample)

            cosim_per_sample = metrics.cosine_similarity(target_cdr_composition, pred_cdr_composition)
            self.cosim_mean.update(cosim_per_sample)

            del target_cdr_composition, pred_cdr_composition

            if not self.stage == 'fit':
                self.recovery_list.append(recovery_acc_per_sample * 100)
                self.tvd_list.append(tvd_per_sample)
                self.cosim_list.append(cosim_per_sample)

        results = {
            'pred_tokens': pred_tokens,
            'names': batch['names'],
            'native': batch['seqs'],
            'recovery': recovery_acc_per_sample,
            'tvd': tvd_per_sample,
            'cosim': cosim_per_sample,
            'sc_tmscores': np.zeros(pred_tokens.shape[0])
        }

        if self.hparams.generator.eval_sc:
            torch.cuda.empty_cache()
            sc_tmscores = self.eval_self_consistency(pred_tokens, batch['coords'], mask=tokens.ne(self.alphabet.padding_idx))
            results['sc_tmscores'] = sc_tmscores

        return results

    def predict_epoch_end(self, results: List[Any]) -> None:
        log_key = 'test' if self.stage == 'test' else 'val'

        acc = self.acc.compute() * 100
        self.acc.reset()
        self.log(f"{log_key}/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        acc_median = torch.median(self.acc_median.compute()) * 100
        self.acc_median.reset()
        self.log(f"{log_key}/acc_median", acc_median, on_step=False, on_epoch=True, prog_bar=True)

        acc_mean = self.acc_mean.compute() * 100
        self.acc_mean.reset()
        self.log(f"{log_key}/acc_mean", acc_mean, on_step=False, on_epoch=True, prog_bar=True)

        tvd_mean = self.tvd_mean.compute()
        self.tvd_mean.reset()
        self.log(f"{log_key}/tvd_mean", tvd_mean, on_step=False, on_epoch=True, prog_bar=True)

        cosim_mean = self.cosim_mean.compute()
        self.cosim_mean.reset()
        self.log(f"{log_key}/cosim_mean", cosim_mean, on_step=False, on_epoch=True, prog_bar=True)

        if self.stage == 'fit':
            self.acc_best.update(acc)
            self.log(f"{log_key}/acc_best", self.acc_best.compute(), on_epoch=True, prog_bar=True)

            self.acc_median_best.update(acc_median)
            self.log(f"{log_key}/acc_median_best", self.acc_median_best.compute(), on_epoch=True, prog_bar=True)

            self.acc_mean_best.update(acc_mean)
            self.log(f"{log_key}/acc_mean_best", self.acc_mean_best.compute(), on_epoch=True, prog_bar=True)

            self.cosim_mean_best.update(cosim_mean)
            self.log(f"{log_key}/cosim_mean_best", self.cosim_mean_best.compute(), on_epoch=True, prog_bar=True)
        else:
            if self.hparams.generator.eval_sc:
                import itertools
                sc_tmscores = list(itertools.chain(*[result['sc_tmscores'] for result in results]))
                self.log(f"{log_key}/sc_tmscores", np.mean(sc_tmscores), on_epoch=True, prog_bar=True)
          
            # save predicted sequences
            self.save_prediction(results, saveto=f'./test_iter{self.hparams.generator.max_iter}_{self.hparams.generator.strategy}_tau{self.hparams.generator.temperature}.fasta')

            # compute standard deviations
            acc_stddev = torch.std(torch.concat(self.recovery_list))
            tvd_stddev = torch.std(torch.concat(self.tvd_list))
            cosim_stddev = torch.std(torch.concat(self.cosim_list))

            # save summary
            summary_dict = {
                'acc': acc,
                'acc_median': acc_median,
                'acc_mean': acc_mean,
                'acc_stddev': acc_stddev,
                'tvd_mean': tvd_mean,
                'tvd_stddev': tvd_stddev,
                'cosim_mean': cosim_mean,
                'cosim_stddev': cosim_stddev,
            }
            self.save_summary(summary_dict, saveto=f'./summary_iter{self.hparams.generator.max_iter}_{self.hparams.generator.strategy}_tau{self.hparams.generator.temperature}.fasta')

    def save_summary(self, summary_dict, saveto=None):
        if saveto:
            saveto = os.path.abspath(saveto)
            # Dynamically construct the summary string
            summary_str = " | ".join([f"{key}={value:.4f}" for key, value in summary_dict.items()])
            
            log.info(f"Saving summary to {saveto}...")  
            with open(saveto, 'w') as fp:
                fp.write(summary_str + "\n")

    def save_prediction(self, results, saveto=None):
        save_dict = {}
        if saveto:
            saveto = os.path.abspath(saveto)
            log.info(f"Saving predictions to {saveto}...")
            fp = open(saveto, 'w')
            fp_native = open('./native.fasta', 'w')

        for entry in results:
            for name, prediction, native, recovery, tvd, cosim, scTM in zip(
                entry['names'],
                self.alphabet.decode(entry['pred_tokens'], remove_special=True),
                entry['native'],
                entry['recovery'],
                entry['tvd'],
                entry['cosim'],
                entry['sc_tmscores'],
            ):
                save_dict[name] = {
                    'prediction': prediction,
                    'native': native,
                    'recovery': recovery
                }
                if saveto:
                    fp.write(f">name={name} | L={len(prediction)} | AAR={recovery:.2f} | TVD={tvd:.2f} | CoSim={cosim:.2f} | scTM={scTM:.2f}\n")
                    fp.write(f"{prediction}\n\n")
                    fp_native.write(f">name={name}\n{native}\n\n")

        if saveto:
            fp.close()
            fp_native.close()
        return save_dict

    def esm_refine(self, pred_ids, only_mask=False):
        """Use ESM-1b to refine model predicted"""
        if not hasattr(self, 'esm'):
            import esm
            self.esm, self.esm_alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
            # self.esm, self.esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            self.esm_batcher = self.esm_alphabet.get_batch_converter()
            self.esm.to(self.device)
            self.esm.eval()

        mask = pred_ids.eq(self.alphabet.mask_idx)

        # _, _, input_ids = self.esm_batcher(
        #     [('_', seq) for seq in decode(pred_ids, self.alphabet)]
        # )
        # decode(pred_ids, self.alphabet)
        # input_ids = convert_by_alphabets(pred_ids, self.alphabet, self.esm_alphabet)

        input_ids = pred_ids
        results = self.esm(
            input_ids.to(self.device), repr_layers=[33], return_contacts=False
        )
        logits = results['logits']
        # refined_ids = logits.argmax(-1)[..., 1:-1]
        refined_ids = logits.argmax(-1)
        refined_ids = convert_by_alphabets(refined_ids, self.esm_alphabet, self.alphabet)

        if only_mask:
            refined_ids = torch.where(mask, refined_ids, pred_ids)
        return refined_ids

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def eval_self_consistency(self, pred_ids, positions, mask=None):
        pred_seqs = self.alphabet.decode(pred_ids, remove_special=True)

        # run_folding:
        sc_tmscores = []
        with torch.no_grad():
            output = self._folding_model.infer(sequences=pred_seqs, num_recycles=4)
            pred_seqs = self.alphabet.decode(output['aatype'], remove_special=True)
            for i in range(positions.shape[0]):
                pred_seq = pred_seqs[i]
                seqlen = len(pred_seq)
                _, sc_tmscore = metrics.calc_tm_score(
                    positions[i, 1:seqlen + 1, :3, :].cpu().numpy(),
                    output['positions'][-1, i, :seqlen, :3, :].cpu().numpy(),
                    pred_seq, pred_seq
                )
                sc_tmscores.append(sc_tmscore)
        return sc_tmscores


def convert_by_alphabets(ids, alphabet1, alphabet2, relpace_unk_to_mask=True):
    sizes = ids.size()
    mapped_flat = ids.new_tensor(
        [alphabet2.get_idx(alphabet1.get_tok(ind)) for ind in ids.flatten().tolist()]
    )
    if relpace_unk_to_mask:
        mapped_flat[mapped_flat.eq(alphabet2.unk_idx)] = alphabet2.mask_idx
    return mapped_flat.reshape(*sizes)
