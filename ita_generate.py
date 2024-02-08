#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import re
import json
import argparse
from time import time
from tqdm import tqdm
from copy import deepcopy

import numpy as np
import torch

from data import AAComplex
from evaluation.rmsd import kabsch
from evaluation import pred_ddg
from utils.logger import print_log
from utils.random_seed import setup_seed
from utils.lr_scheduler import get_scheduler
from datasets.data_utils import Alphabet

from ita_train import get_config, prepare_efficient_mc_att

from ASG import ASG 
from data import VOCAB

ALPHABET = Alphabet(**{'name': 'esm', 'featurizer': 'antibody'})

def set_cdr(cplx, seq, x, cdr='H3'):
    cdr = cdr.upper()
    cplx: AAComplex = deepcopy(cplx)
    chains = cplx.peptides
    cdr_chain_key = cplx.heavy_chain if 'H' in cdr else cplx.light_chain
    refined_chain = chains[cdr_chain_key]
    start, end = cplx.get_cdr_pos(cdr)
    start_pos, end_pos = refined_chain.get_ca_pos(start), refined_chain.get_ca_pos(end)
    start_trans, end_trans = x[0][1] - start_pos, x[-1][1] - end_pos
    # left to start of cdr
    for i in range(0, start):
        refined_chain.set_residue_translation(i, start_trans)
    # end of cdr to right
    for i in range(end + 1, len(refined_chain)):
        refined_chain.set_residue_translation(i, end_trans)
    # cdr 
    for i, residue_x, symbol in zip(range(start, end + 1), x, seq):
        center = residue_x[4] if len(residue_x) > 4 else None
        refined_chain.set_residue(i, symbol,
            {
                'N': residue_x[0],
                'CA': residue_x[1],
                'C': residue_x[2],
                'O': residue_x[3]
            }, center, gen_side_chain=False
        )
    new_cplx = AAComplex(cplx.pdb_id, chains, cplx.heavy_chain, cplx.light_chain,
                         cplx.antigen_chains, numbering=None, cdr_pos=cplx.cdr_pos,
                         skip_cal_interface=True)
    return new_cplx

def parse():
    parser = argparse.ArgumentParser(description='ITA generation')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Path to the checkpoint')
    parser.add_argument('--esm_ckpt', type=str, required=True, help='Path to ESM checkpoint')
    parser.add_argument('--esm_type', type=str, required=True)

    parser.add_argument('--test_set', type=str, required=True,
                        help='Path to test set (antibodies to be optimized)')
    parser.add_argument('--n_samples', type=int, default=100, help='Number of samples')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='GPU to use, -1 for cpu')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Path to the output files, default under the ckpt directory')
    return parser.parse_args()


def main(args):
    print(str(args))
    mode = get_config(args.ckpt)
    print(f'mode: {mode}')
    # model = torch.load(args.ckpt, map_location='cpu')
    device = torch.device('cpu' if args.gpu == -1 else f'cuda:{args.gpu}')
    # model.to(device)

    # load esm
    esm_kwargs = {
        'esm_type': args.esm_type,
        'temp': 1,
        'new_load': False
    }
    model = ASG(args.esm_ckpt, args.ckpt, device, **esm_kwargs)

    dataset, _ = prepare_efficient_mc_att(model, mode, args.test_set, 32)
    if args.save_dir is None:
        args.save_dir = os.path.split(args.ckpt)[0]
    args.save_dir = os.path.join(args.save_dir, 'ita_results')
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # writing original structrues
    origin_cplx = [dataset.data[i] for i in dataset.idx_mapping]
    origin_cplx_paths = []
    out_dir = os.path.join(args.save_dir, 'original')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print_log(f'Writing original structures to {out_dir}')
    for cplx in tqdm(origin_cplx):
        pdb_path = os.path.join(out_dir, cplx.get_id() + '.pdb')
        cplx.to_pdb(pdb_path)
        origin_cplx_paths.append(os.path.abspath(pdb_path))

    log = open(os.path.join(args.save_dir, 'log.txt'), 'w')
    res_dir = os.path.join(args.save_dir, 'optimized')
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    scores = []
    for i in tqdm(range(len(dataset))):
        origin_input = dataset[i]
        inputs = [origin_input for _ in range(args.n_samples)]
        cur_scores, results = [], []
        with torch.no_grad():
            batch = dataset.collate_fn(inputs)
            ppls, seqs, xs, true_xs, aligned = model.infer(batch, device, greedy=False)
            results.extend([(seqs[i], xs[i], true_xs[i], aligned) for i in range(len(seqs))])
        for n, (seq, x, true_x, aligned) in enumerate(results):
            if not aligned:
                ca_aligned, rotation, t = kabsch(x[:, 1, :], true_x[:, 1, :])
                x = np.dot(x - np.mean(x, axis=0), rotation) + t
            new_cplx = set_cdr(origin_cplx[i], seq, x, cdr='H' + str(model.mc_att.cdr_type))
            pdb_path = os.path.join(res_dir, new_cplx.get_id() + f'_{n}.pdb')
            new_cplx.to_pdb(pdb_path)
            new_cplx = AAComplex(
                new_cplx.pdb_id, new_cplx.peptides,
                new_cplx.heavy_chain, new_cplx.light_chain,
                new_cplx.antigen_chains)
            try:
                score = pred_ddg(origin_cplx_paths[i], os.path.abspath(pdb_path))
            except Exception as e:
                print_log(str(e), level='ERROR')
                score = 0
            cur_scores.append(score)
        mean_score = np.mean(cur_scores)
        best_score_idx = min([k for k in range(len(cur_scores))], key=lambda k: cur_scores[k])
        scores.append(cur_scores[best_score_idx])
        log.write(f'pdb {origin_cplx[i].get_id()}, mean ddg {mean_score}, best ddg {cur_scores[best_score_idx]}, sample {best_score_idx}\n')
        log.flush()
    mean_score = np.mean(scores)
    log_line = f'overall ddg mean {mean_score} WITHOUT sidechain packing'
    print_log(log_line)
    log.write(log_line + '\n')
    log.close()


if __name__ == '__main__':
    args = parse()
    setup_seed(args.seed)
    main(args)
