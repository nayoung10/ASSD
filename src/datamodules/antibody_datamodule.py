import os 
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from src import utils
from src.datamodules import register_datamodule
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from .datasets.data_utils import Alphabet, MaxTokensBatchSampler
from .datasets.antibody import EquiAACDataset

log = utils.get_logger(__name__)


# @register_datamodule('struct2seq')
@register_datamodule('equiaac')
class EquiAACDataModule(LightningDataModule):

    def __init__(
        self,
        data_dir: str = "data/cdrh3/",
        max_length: int = 500,
        atoms: List[str] = ('N', 'CA', 'C', 'O'),
        alphabet=None,
        batch_size: int = 64,
        max_tokens: int = 6000,
        sort: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
        test_split='test'
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        self.alphabet = None

        self.train_data: Optional[Dataset] = None
        self.valid_data: Optional[Dataset] = None
        self.test_data: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`.
        """

        if stage == 'fit' or stage is None:
            self.train_dataset = EquiAACDataset(os.path.join(self.hparams.data_dir, "train.json"))
            self.valid_dataset = EquiAACDataset(os.path.join(self.hparams.data_dir, "valid.json"))
        
        if stage == 'test' or stage == 'predict' or stage is None:
            self.test_dataset = EquiAACDataset(os.path.join(self.hparams.data_dir, self.hparams.test_split + ".json"))

        self.alphabet = Alphabet(**self.hparams.alphabet)
        self.collate_batch = self.alphabet.featurizer

    def _build_batch_sampler(self, dataset, max_tokens, shuffle=False, distributed=True):
        is_distributed = distributed and torch.distributed.is_initialized()

        batch_sampler = MaxTokensBatchSampler(
            dataset=dataset,
            shuffle=shuffle,
            distributed=is_distributed,
            batch_size=self.hparams.batch_size,
            max_tokens=max_tokens,
            sort=self.hparams.sort,
            drop_last=False,
            sort_key=lambda i: len(dataset[i]['seq']))
        return batch_sampler

    def train_dataloader(self):
        if not hasattr(self, 'train_batch_sampler'):
            self.train_batch_sampler = self._build_batch_sampler(
                self.train_dataset,
                max_tokens=self.hparams.max_tokens,
                shuffle=True
            )
        return DataLoader(
            dataset=self.train_dataset,
            batch_sampler=self.train_batch_sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate_batch
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_dataset,
            batch_sampler=self._build_batch_sampler(
                self.valid_dataset, max_tokens=self.hparams.max_tokens, distributed=False),
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate_batch
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_sampler=self._build_batch_sampler(
                self.test_dataset, max_tokens=self.hparams.max_tokens, distributed=False),
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate_batch
        )
