import numpy as np, pandas as pd
EDGES = np.array([0.125, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0])   # nats; 11 bins
CONTRASTS = {"morph/int": ("morph_onset", "internal", "ptype"), "word/int": ("word_onset", "internal", "ptype"), "fused/split": ("CROSS", "WITHIN_SPLIT", "aclass")}

def ratio(d, num, den, col, key):
    """Direct standardization over cells of `key` (array aligned with d's RangeIndex);
    weights = all aligned positions in cells where both classes occur. Returns (ratio, coverage)."""
    d = d[d[col] != "excluded"] if col == "ptype" else d[d[col].isin(["CROSS", "WITHIN_SPLIT", "EXACT"])]
    k = key[d.index.values] if key is not None else np.zeros(len(d), np.int64)
    g = pd.DataFrame({"k": k, "rej": d.rej.values, "c": d[col].values})
    w = g.groupby("k").size()
    a = g[g.c == num].groupby("k").rej.mean(); b = g[g.c == den].groupby("k").rej.mean()
    common = a.index.intersection(b.index); ww = w.loc[common] / w.loc[common].sum()
    den_v = (b.loc[common] * ww).sum()
    return (float((a.loc[common] * ww).sum() / den_v) if den_v > 0 else float("nan")), float(w.loc[common].sum() / w.sum())

def keys_for(df):
    qT = pd.qcut(df.mT.rank(method="first"), 10, labels=False).values; qD = pd.qcut(df.mD.rank(method="first"), 10, labels=False).values
    aT = np.digitize(df.mT.values, EDGES); aD = np.digitize(df.mD.values, EDGES)
    return {"raw": None, "target deciles": qT, "target abs bins": aT, "draft deciles": qD, "draft abs bins": aD,
            "both deciles": qT * 10 + qD, "both abs bins": aT * 11 + aD}

def boot_ci(d, num, den, col, key, rng, reps=500):
    """95% document-bootstrap interval of `ratio` for a fixed cell key (e.g. absolute margin bins)."""
    d = d[d[col] != "excluded"] if col == "ptype" else d[d[col].isin(["CROSS", "WITHIN_SPLIT", "EXACT"])]
    k = key[d.index.values]; cells, kc = np.unique(k, return_inverse=True); docs, dc = np.unique(d.doc.values, return_inverse=True)
    C = len(cells); flat = dc * C + kc; shape = (len(docs), C); rej = d.rej.values.astype(float); c = d[col].values
    def mat(mask, wts=None):
        return np.bincount(flat[mask], weights=None if wts is None else wts[mask], minlength=shape[0] * C).reshape(shape).astype(float)
    every = np.ones(len(d), bool); isn = c == num; isd = c == den
    N, Nn, Rn, Nd, Rd = mat(every), mat(isn), mat(isn, rej), mat(isd), mat(isd, rej)
    vals = []
    for _ in range(reps):
        w = np.bincount(rng.integers(0, len(docs), len(docs)), minlength=len(docs)).astype(float)
        n, nn, rn, nd, rd = w @ N, w @ Nn, w @ Rn, w @ Nd, w @ Rd
        ok = (nn > 0) & (nd > 0); ww = n[ok] / n[ok].sum(); dv = (rd[ok] / nd[ok] * ww).sum()
        vals.append((rn[ok] / nn[ok] * ww).sum() / dv if dv > 0 else np.nan)
    return [round(float(np.nanquantile(vals, .025)), 2), round(float(np.nanquantile(vals, .975)), 2)]
