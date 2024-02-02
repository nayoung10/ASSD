import math
import torch
import torch.nn.functional as F
from functools import partial
import numpy as np


def luost_rmsd(res_list1: list, res_list2: list):
    res_short, res_long = (res_list1, res_list1) if len(res_list1) < len(res_list2) else (res_list2, res_list1)
    M, N = len(res_short), len(res_long)

    def d(i, j):
        coord_i = res_short[i]
        coord_j = res_long[j]
        return ((coord_i - coord_j) ** 2).sum()

    SD = np.full([M, N], np.inf)
    for i in range(M):
        j = N - (M - i)
        SD[i, j] = sum([d(i + k, j + k) for k in range(N - j)])

    for j in range(N):
        SD[M - 1, j] = d(M - 1, j)

    for i in range(M - 2, -1, -1):
        for j in range((N - (M - i)) - 1, -1, -1):
            SD[i, j] = min(
                d(i, j) + SD[i + 1, j + 1],
                SD[i, j + 1]
            )

    min_SD = SD[0, :N - M + 1].min()
    best_RMSD = np.sqrt(min_SD / M)
    return best_RMSD


def rmsd(pred, target, mask=None):
    assert pred.shape == target.shape
    if mask is None:
        mask = torch.ones_like(pred, dtype=torch.bool)

    rmsd = []
    for p, t, m in zip(pred, target, mask):
        rmsd.append(luost_rmsd(p[m], t[m]))
    return np.mean(rmsd)


def accuracy(pred, target, mask=None, reduction='all'):
    assert pred.shape == target.shape
    if mask is None:
        mask = torch.ones_like(pred, dtype=torch.bool)
    
    return (pred[mask] == target[mask]).sum() / mask.sum()

def accuracy_per_sample(pred, target, mask=None):
    assert pred.shape == target.shape
    bsz = target.shape[0] 

    if mask is None:
        mask = torch.ones_like(pred, dtype=torch.bool)

    pred = pred.view(bsz, -1)
    target = target.view(bsz, -1)
    mask = mask.view(bsz, -1)

    return ((pred == target) * mask).sum(1) / mask.sum(1)


from tmtools import tm_align

def calc_tm_score(pos_1, pos_2, seq_1, seq_2):
    tm_results = tm_align(np.float64(pos_1), np.float64(pos_2), seq_1, seq_2)
    return tm_results.tm_norm_chain1, tm_results.tm_norm_chain2 


def get_cdr_composition(tokens, cdr_mask, alphabet):
    """
    Calculates the CDR composition for each sequence in the batch.
    
    :param tokens: PyTorch tensor of shape [N, max_len] containing token indices
    :param cdr_mask: PyTorch tensor of shape [N, max_len] with True for CDR regions
    :param alphabet: An object of class 'alphabet'
    
    :return: cdr_composition tensor of shape [N, vocab_size]
    """
    # Get the vocabulary size
    vocab_size = len(alphabet)
    
    # One-hot encode the tokens
    one_hot_tokens = torch.zeros(tokens.size(0), tokens.size(1), vocab_size, dtype=tokens.dtype, device=tokens.device)
    one_hot_tokens.scatter_(2, tokens.unsqueeze(-1), 1)
    
    # Apply cdr_mask to one-hot tokens (this operation will convert it to float)
    masked_tokens = one_hot_tokens.float() * cdr_mask.unsqueeze(-1).float()
    
    # Sum over the sequence length to get the composition
    cdr_comp = masked_tokens.sum(dim=1)
    
    # Cast the result back to original dtype
    cdr_comp = cdr_comp.to(tokens.dtype)
    
    return cdr_comp

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two batches of vectors.
    
    Parameters:
    - a : torch.Tensor of shape (batch_size, vocab_size)
    - b : torch.Tensor of shape (batch_size, vocab_size)

    Returns:
    - cosine_sim : torch.Tensor of shape (batch_size,)
    """

    # convert to float
    a = a.float()
    b = b.float()
    
    # Normalize vectors in the feature dimension
    a_normalized = F.normalize(a, p=2, dim=1)
    b_normalized = F.normalize(b, p=2, dim=1)
    
    # Compute cosine similarity
    cosine_sim = torch.sum(a_normalized * b_normalized, dim=1)

    return cosine_sim

def total_variation_distance(p, q):
    """
    Compute total variation distance between two batches of distributions.
    
    Parameters:
    - p : torch.Tensor of shape (batch_size, vocab_size)
    - q : torch.Tensor of shape (batch_size, vocab_size)

    Returns:
    - tv_distance : torch.Tensor of shape (batch_size,)
    """

    # convert to float
    p = p.float()
    q = q.float()

    # Normalize (if necessary)
    p = p / p.sum(dim=1, keepdim=True)
    q = q / q.sum(dim=1, keepdim=True)
    
    tv_distance = 0.5 * torch.sum(torch.abs(p - q), dim=1)
    return tv_distance
