import json
import os
import pickle 
from tqdm import tqdm
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from src import utils
from torch.nn import functional as F
from torch.utils.data.datapipes.map import SequenceWrapper
from torch.utils.data.dataset import Subset

from .data_utils import Alphabet
from src.datamodules.datasets.pdb_utils import AAComplex, Protein, VOCAB

import esm

log = utils.get_logger(__name__)

class EquiAACAntigenDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, save_dir=None, num_entry_per_file=-1, random=False, ctx_cutoff=8.0, interface_cutoff=12.0):
        '''
        file_path: path to the dataset
        save_dir: directory to save the processed data
        num_entry_per_file: number of entries in a single file. -1 to save all data into one file 
                            (In-memory dataset)
        '''
        super().__init__()
        if save_dir is None:
            if not os.path.isdir(file_path):
                save_dir = os.path.split(file_path)[0]
            else:
                save_dir = file_path
            prefix = os.path.split(file_path)[1]
            if '.' in prefix:
                prefix = prefix.split('.')[0]
            save_dir = os.path.join(save_dir, f'{prefix}_processed_antigen')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        metainfo_file = os.path.join(save_dir, '_metainfo')
        self.data: List[AAComplex] = []  # list of ABComplex

        # try loading preprocessed files
        need_process = False
        try:
            with open(metainfo_file, 'r') as fin:
                metainfo = json.load(fin)
                self.num_entry = metainfo['num_entry']
                self.file_names = metainfo['file_names']
                self.file_num_entries = metainfo['file_num_entries']
        except Exception as e:
            utils.print_log(f'Faild to load file {metainfo_file}, error: {e}', level='WARN')
            need_process = True

        if need_process:
            # preprocess
            self.file_names, self.file_num_entries = [], []
            self.preprocess(file_path, save_dir, num_entry_per_file)
            self.num_entry = sum(self.file_num_entries)

            metainfo = {
                'num_entry': self.num_entry,
                'file_names': self.file_names,
                'file_num_entries': self.file_num_entries
            }
            with open(metainfo_file, 'w') as fout:
                json.dump(metainfo, fout)

        self.random = random
        self.cur_file_idx, self.cur_idx_range = 0, (0, self.file_num_entries[0])  # left close, right open
        self._load_part()

        # user defined variables
        self.idx_mapping = [i for i in range(self.num_entry)]
        self.mode = '111'  # H/L/Antigen, 1 for include, 0 for exclude
        self.ctx_cutoff = ctx_cutoff
        self.interface_cutoff = interface_cutoff

    def _save_part(self, save_dir, num_entry):
        file_name = os.path.join(save_dir, f'part_{len(self.file_names)}.pkl')
        utils.print_log(f'Saving {file_name} ...')
        file_name = os.path.abspath(file_name)
        if num_entry == -1:
            end = len(self.data)
        else:
            end = min(num_entry, len(self.data))
        with open(file_name, 'wb') as fout:
            pickle.dump(self.data[:end], fout)
        self.file_names.append(file_name)
        self.file_num_entries.append(end)
        self.data = self.data[end:]

    def _load_part(self):
        f = self.file_names[self.cur_file_idx]
        utils.print_log(f'Loading preprocessed file {f}, {self.cur_file_idx + 1}/{len(self.file_names)}')
        with open(f, 'rb') as fin:
            del self.data
            self.data = pickle.load(fin)
        self.access_idx = [i for i in range(len(self.data))]
        if self.random:
            np.random.shuffle(self.access_idx)

    def _check_load_part(self, idx):
        if idx < self.cur_idx_range[0]:
            while idx < self.cur_idx_range[0]:
                end = self.cur_idx_range[0]
                self.cur_file_idx -= 1
                start = end - self.file_num_entries[self.cur_file_idx]
                self.cur_idx_range = (start, end)
            self._load_part()
        elif idx >= self.cur_idx_range[1]:
            while idx >= self.cur_idx_range[1]:
                start = self.cur_idx_range[1]
                self.cur_file_idx += 1
                end = start + self.file_num_entries[self.cur_file_idx]
                self.cur_idx_range = (start, end)
            self._load_part()
        idx = self.access_idx[idx - self.cur_idx_range[0]]
        return idx
     
    ########### load data from file_path and add to self.data ##########
    def preprocess(self, file_path, save_dir, num_entry_per_file):
        '''
        Load data from file_path and add processed data entries to self.data.
        Remember to call self._save_data(num_entry_per_file) to control the number
        of items in self.data (this function will save the first num_entry_per_file
        data and release them from self.data) e.g. call it when len(self.data) reaches
        num_entry_per_file.
        '''
        with open(file_path, 'r') as fin:
            lines = fin.read().strip().split('\n')
        for line in tqdm(lines):
            item = json.loads(line)
            try:
                protein = Protein.from_pdb(item['pdb_data_path'])
            except AssertionError as e:
                utils.print_log(e, level='ERROR')
                utils.print_log(f'parse {item["pdb"]} pdb failed, skip', level='ERROR')
                continue

            pdb_id, peptides = item['pdb'], protein.peptides
            complex_obj = AAComplex(pdb_id, peptides, item['heavy_chain'], item['light_chain'], item['antigen_chains'])
            processed_item = self._process_item(complex_obj)
            self.data.append(processed_item)
            if num_entry_per_file > 0 and len(self.data) >= num_entry_per_file:
                self._save_part(save_dir, num_entry_per_file)
        if len(self.data):
            self._save_part(save_dir, num_entry_per_file)

    def _process_item(self, item):
        hc, lc = item.get_heavy_chain(), item.get_light_chain()
        antigen_chains = item.get_antigen_chains(interface_only=False, cdr=None)

        # Process the antigen chains only
        coords = {
            'N': [],
            'CA': [],
            'C': [],
            'O': []
        }

        for chain in antigen_chains:
            for i in range(len(chain)):
                residue = chain.get_residue(i)
                coord_map = residue.get_coord_map()

                for atom in ['N', 'CA', 'C', 'O']:
                    if atom in coord_map:
                        coords[atom].append(coord_map[atom])
                    else:
                        coords[atom].append((0, 0, 0))

        # Convert coordinates into tensors
        for atom in coords:
            coords[atom] = torch.tensor(np.array(coords[atom]), dtype=torch.float)

        # Initialize L to be the length of the heavy chain with zeros (indicating non-CDR residues)
        L = ['0'] * len(hc)

        # Set CDR positions for the heavy chain
        for i in range(1, 4):  # Assuming there are 3 CDRs
            begin, end = item.get_cdr_pos(f'H{i}')
            for pos in range(begin, end + 1):
                L[pos] = str(i)
        
        # Combine the antigen chains into a single sequence
        antigen_seq = '-'.join([chain.get_seq() for chain in antigen_chains]) if len(antigen_chains) > 1 else antigen_chains[0].get_seq()
        # Remove unknown sequence
        antigen_seq = antigen_seq.replace('*', '')

        res = {
            'seq': item.get_heavy_chain().get_seq() + '.' + antigen_seq,
            'coords': coords,
            'cdr': ''.join(L),
            'name': item.pdb_id
        }
        return res  
        
    def __getitem__(self, idx):
        idx = self.idx_mapping[idx]
        idx = self._check_load_part(idx)
        processed_item = self.data[idx]

        # The following step may not be necessary if your _save_part and 
        # _check_load_part functions already ensure that data is in the desired format.
        # However, if there's a chance that some items might not be preprocessed, 
        # you could use the following conditional check:
        # 
        if not isinstance(processed_item, dict): 
            processed_item = self._process_item(processed_item)
            self.data[idx] = processed_item

        return processed_item

    def __len__(self):
        return self.num_entry


# NOTE: batch is a list mapping
# each mapping has columns:
#   name: str
#   seq: str. sequence of amino acids
#   coords: Dict[str, List[1d-array]]). e.g., {"N": [[0, 0, 0], [0.1, 0.1, 0.1], ..], "Ca": [...], ..}
def collate_batch(
    batch: List[Dict[str, Any]],
    batch_converter,
    transform=None,
    atoms=('N', 'CA', 'C', 'O')
):
    seqs, coords = [], []
    names = []
    for entry in batch:
        _seq, _coords = entry['seq'], entry['coords']
        seqs.append(_seq)
        # [L, 3] x 4 -> [L, 4, 3]
        coords.append(
            # np.stack([_coords[c] for c in ['N', 'CA', 'C', 'O']], 1)
            np.stack([_coords[c] for c in atoms], 1)
        )
        names.append(entry['name'])

    coords, confidence, strs, tokens, lengths, coord_mask = batch_converter.from_lists(
        coords_list=coords, confidence_list=None, seq_list=seqs
    )

    # coords, tokens, coord_mask, lengths = featurize(batch, torch.device('cpu'), 0)
    # coord_mask = coord_mask > 0.5
    batch_data = {
        'coords': coords,
        'tokens': tokens,
        'confidence': confidence,
        'coord_mask': coord_mask,
        'lengths': lengths,
        'seqs': seqs,
        'names': names
    }

    if transform is not None:
        batch_data = transform(batch_data)

    return batch_data


class CoordBatchConverter(esm.data.BatchConverter):
    def __init__(self, alphabet, coord_pad_inf=False, coord_nan_to_zero=True, to_pifold_format=False):
        super().__init__(alphabet)
        self.coord_pad_inf = coord_pad_inf
        self.to_pifold_format = to_pifold_format
        self.coord_nan_to_zero = coord_nan_to_zero

    def __call__(self, raw_batch: Sequence[Tuple[Sequence, str]], device=None):
        """
        Args:
            raw_batch: List of tuples (coords, confidence, seq)
            In each tuple,
                coords: list of floats, shape L x n_atoms x 3
                confidence: list of floats, shape L; or scalar float; or None
                seq: string of length L
        Returns:
            coords: Tensor of shape batch_size x L x n_atoms x 3
            confidence: Tensor of shape batch_size x L
            strs: list of strings
            tokens: LongTensor of shape batch_size x L
            padding_mask: ByteTensor of shape batch_size x L
        """
        # self.alphabet.cls_idx = self.alphabet.get_idx("<cath>")
        batch = []
        for coords, confidence, seq in raw_batch:
            if confidence is None:
                confidence = 1.
            if isinstance(confidence, float) or isinstance(confidence, int):
                confidence = [float(confidence)] * len(coords)
            if seq is None:
                seq = 'X' * len(coords)
            batch.append(((coords, confidence), seq))

        coords_and_confidence, strs, tokens = super().__call__(batch)

        if self.coord_pad_inf:
            # pad beginning and end of each protein due to legacy reasons
            coords = [
                F.pad(torch.tensor(cd), (0, 0, 0, 0, 1, 1), value=np.nan)
                for cd, _ in coords_and_confidence
            ]
            confidence = [
                F.pad(torch.tensor(cf), (1, 1), value=-1.)
                for _, cf in coords_and_confidence
            ]
        else:
            coords = [
                torch.tensor(cd) for cd, _ in coords_and_confidence
            ]
            confidence = [
                torch.tensor(cf) for _, cf in coords_and_confidence
            ]
        # coords = self.collate_dense_tensors(coords, pad_v=np.nan)
        coords = self.collate_dense_tensors(coords, pad_v=np.nan)
        confidence = self.collate_dense_tensors(confidence, pad_v=-1.)

        if self.to_pifold_format:
            coords, tokens, confidence = ToPiFoldFormat(X=coords, S=tokens, cfd=confidence)

        lengths = tokens.ne(self.alphabet.padding_idx).sum(1).long()
        if device is not None:
            coords = coords.to(device)
            confidence = confidence.to(device)
            tokens = tokens.to(device)
            lengths = lengths.to(device)

        coord_padding_mask = torch.isnan(coords[:, :, 0, 0])
        coord_mask = torch.isfinite(coords.sum([-2, -1]))
        confidence = confidence * coord_mask + (-1.) * coord_padding_mask

        if self.coord_nan_to_zero:
            coords[torch.isnan(coords)] = 0.

        return coords, confidence, strs, tokens, lengths, coord_mask

    def from_lists(self, coords_list, confidence_list=None, seq_list=None, device=None):
        """
        Args:
            coords_list: list of length batch_size, each item is a list of
            floats in shape L x 3 x 3 to describe a backbone
            confidence_list: one of
                - None, default to highest confidence
                - list of length batch_size, each item is a scalar
                - list of length batch_size, each item is a list of floats of
                    length L to describe the confidence scores for the backbone
                    with values between 0. and 1.
            seq_list: either None or a list of strings
        Returns:
            coords: Tensor of shape batch_size x L x 3 x 3
            confidence: Tensor of shape batch_size x L
            strs: list of strings
            tokens: LongTensor of shape batch_size x L
            padding_mask: ByteTensor of shape batch_size x L
        """
        batch_size = len(coords_list)
        if confidence_list is None:
            confidence_list = [None] * batch_size
        if seq_list is None:
            seq_list = [None] * batch_size
        raw_batch = zip(coords_list, confidence_list, seq_list)
        return self.__call__(raw_batch, device)

    @staticmethod
    def collate_dense_tensors(samples, pad_v):
        """
        Takes a list of tensors with the following dimensions:
            [(d_11,       ...,           d_1K),
             (d_21,       ...,           d_2K),
             ...,
             (d_N1,       ...,           d_NK)]
        and stack + pads them into a single tensor of:
        (N, max_i=1,N { d_i1 }, ..., max_i=1,N {diK})
        """
        if len(samples) == 0:
            return torch.Tensor()
        if len(set(x.dim() for x in samples)) != 1:
            raise RuntimeError(
                f"Samples has varying dimensions: {[x.dim() for x in samples]}"
            )
        (device,) = tuple(set(x.device for x in samples))  # assumes all on same device
        max_shape = [max(lst) for lst in zip(*[x.shape for x in samples])]
        result = torch.empty(
            len(samples), *max_shape, dtype=samples[0].dtype, device=device
        )
        result.fill_(pad_v)
        for i in range(len(samples)):
            result_i = result[i]
            t = samples[i]
            result_i[tuple(slice(0, k) for k in t.shape)] = t
        return result


def new_arange(x, *size):
    """
    Return a Tensor of `size` filled with a range function on the device of x.
    If size is empty, using the size of the variable x.
    """
    if len(size) == 0:
        size = x.size()
    return torch.arange(size[-1], device=x.device).expand(*size).contiguous()


class ToSabdabDataFormat(object):
    def __init__(self, alphabet) -> None:
        self.alphabet_ori = alphabet

        from src.utils.protein import constants
        UNK = constants.ressymb_to_resindex['X']
        self.aa_map = {}
        for ind, tok in enumerate(alphabet.all_toks):
            if tok != '<pad>':
                self.aa_map[ind] = constants.ressymb_to_resindex.get(tok, UNK)
            else:
                self.aa_map[ind] = 21

    def _map_aatypes(self, tokens):
        sizes = tokens.size()
        mapped_aa_flat = tokens.new_tensor([self.aa_map[ind] for ind in tokens.flatten().tolist()])
        return mapped_aa_flat.reshape(*sizes)

    def __call__(self, batch_data) -> Any:
        """
            coords          -> `pos_heavyatom` [B, num_res, num_atom, 3]
            tokens          -> `aa` [B, num_res]
            coord_mask      -> `mask_heavyatom` [B, num_res, num_atom]
            all_zeros       -> `mask` [B, num_res]
            all_zeros       -> `chain_nb` [B, num_res]
            range           -> `res_nb` [B, num_res]
            coord_mask      -> `generate_flag` [B, num_res]
            all_ones        -> `fragment_type` [B, num_res]

            coord_padding_mask: coord_padding_mask
            confidence: confidence,
        """

        batch_data['pos_heavyatom'] = batch_data.pop('coords')
        batch_data['aa'] = self._map_aatypes(batch_data.pop('tokens'))
        batch_data['mask'] = batch_data.pop('coord_mask').bool()
        batch_data['mask_heavyatom'] = batch_data['mask'][:, :, None].repeat(1, 1, batch_data['pos_heavyatom'].shape[2])
        batch_data['chain_nb'] = torch.full_like(batch_data['aa'], fill_value=0, dtype=torch.int64)
        batch_data['res_nb'] = new_arange(batch_data['aa'])
        batch_data['generate_flag'] = batch_data['mask'].clone()
        batch_data['fragment_type'] = torch.full_like(batch_data['aa'], fill_value=1, dtype=torch.int64)

        return batch_data


def ToPiFoldFormat(X, S, cfd, pad_special_tokens=False):
    mask = torch.isfinite(torch.sum(X, [-2, -1]))  # atom mask
    numbers = torch.sum(mask, dim=1).long()

    S_new = torch.zeros_like(S)
    X_new = torch.zeros_like(X) + np.nan
    cfd_new = torch.zeros_like(cfd)

    for i, n in enumerate(numbers):
        X_new[i, :n] = X[i][mask[i] == 1]
        S_new[i, :n] = S[i][mask[i] == 1]
        cfd_new[i, :n] = cfd[i][mask[i] == 1]

    X = X_new
    S = S_new
    cfd = cfd_new

    return X, S, cfd_new


class Featurizer(object):
    def __init__(self, alphabet: Alphabet, 
                 to_pifold_format=False, 
                 coord_nan_to_zero=True,
                 atoms=('N', 'CA', 'C', 'O')):
        self.alphabet = alphabet
        self.batcher = CoordBatchConverter(
            alphabet=alphabet,
            coord_pad_inf=alphabet.add_special_tokens,
            to_pifold_format=to_pifold_format, 
            coord_nan_to_zero=coord_nan_to_zero
        )

        self.atoms = atoms

    def __call__(self, raw_batch: dict):
        seqs, coords, names, cdrs = [], [], [], []  # Step 1: Initialize cdrs list
        for entry in raw_batch:
            # [L, 3] x 4 -> [L, 4, 3]
            if isinstance(entry['coords'], dict):
                coords.append(np.stack([entry['coords'][atom] for atom in self.atoms], 1))
            else:
                coords.append(entry['coords'])
            seqs.append(entry['seq']) 
            names.append(entry['name'])
            cdrs.append(torch.tensor([int(c) for c in entry['cdr']]))  # Convert 'cdr' string to tensor

        coords, confidence, strs, tokens, lengths, coord_mask = self.batcher.from_lists(
            coords_list=coords, confidence_list=None, seq_list=seqs
        )

        # Initialize the cdr_tensor with zeros
        cdr_tensor = torch.zeros_like(tokens)
        
        # Determine the start index based on whether a BOS token is prepended
        start_idx = 1 if self.alphabet.prepend_bos else 0

        # Fill in the values from the cdr list
        for i, cdr in enumerate(cdrs):
            
            # Set the cdr values to the corresponding positions
            cdr_tensor[i, start_idx:start_idx + len(cdr)] = cdr[:len(cdr)]

        batch = {
            'coords': coords,
            'tokens': tokens,
            'confidence': confidence,
            'coord_mask': coord_mask,
            'lengths': lengths,
            'seqs': seqs,
            'names': names,
            'cdrs': cdr_tensor  # Add 'cdr' tensor to the batch
        }
        return batch
