import numpy as np
from numba import njit

@njit(cache=True)
def simulate(match, stop, starts, kmax):
    """Greedy SD replay. Block drafts until stop[t] (inclusive), kmax tokens, or end of text.
    Verification keeps the longest matching prefix; +1 correction/bonus token from the target.
    Returns per-document (calls, drafted, verified_rejections)."""
    nd = len(starts) - 1
    calls = np.zeros(nd, np.int64); drafted = np.zeros(nd, np.int64); rej = np.zeros(nd, np.int64)
    for d in range(nd):
        s, e = starts[d], starts[d + 1]
        c = s
        while c < e:
            j = 0; first_mis = -1
            while True:
                p = c + j
                if p >= e: break
                j += 1
                if first_mis < 0 and not match[p]: first_mis = p
                if stop[p] or j >= kmax: break
            calls[d] += 1; drafted[d] += j
            if first_mis >= 0:
                rej[d] += 1; c = first_mis + 1
            else:
                c = c + j + 1
    return calls, drafted, rej

@njit(cache=True)
def simulate_oracle(match, starts, kmax):
    """Oracle: drafts exactly up to and excluding the first mismatch (max kmax)."""
    nd = len(starts) - 1
    calls = np.zeros(nd, np.int64); drafted = np.zeros(nd, np.int64)
    for d in range(nd):
        s, e = starts[d], starts[d + 1]
        c = s
        while c < e:
            j = 0
            while c + j < e and j < kmax and match[c + j]: j += 1
            calls[d] += 1; drafted[d] += j
            c = c + j + 1
    return calls, drafted

@njit(cache=True)
def simulate_cum(match, lp, starts, kmax, thr):
    """Stop when cumulative draft log-prob of the block falls below thr (joint-confidence rule)."""
    nd = len(starts) - 1
    calls = np.zeros(nd, np.int64); drafted = np.zeros(nd, np.int64)
    for d in range(nd):
        s, e = starts[d], starts[d + 1]
        c = s
        while c < e:
            j = 0; first_mis = -1; acc = 0.0
            while True:
                p = c + j
                if p >= e: break
                j += 1
                acc += lp[p]
                if first_mis < 0 and not match[p]: first_mis = p
                if acc < thr or j >= kmax: break
            calls[d] += 1; drafted[d] += j
            if first_mis >= 0: c = first_mis + 1
            else: c = c + j + 1
    return calls, drafted

def doc_starts(doc):
    idx = np.flatnonzero(np.r_[True, doc[1:] != doc[:-1]])
    return np.r_[idx, len(doc)].astype(np.int64)
