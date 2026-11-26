"""Helpers for rebuttal analyses on released teacher-forced logs (CPU only)."""
import glob, os
from pathlib import Path
import numpy as np, pandas as pd

# scripts/rebuttal/cpu/ksdlib.py -> repo root is three levels up; override with KSD_REPO.
REPO = Path(os.environ.get("KSD_REPO", Path(__file__).resolve().parents[3]))
ROOT = str(REPO / "results" / "naacl_experiments")   # frozen inputs, read-only
OUT = REPO / "results" / "rebuttal" / "cpu"            # all outputs of these scripts
OUT.mkdir(parents=True, exist_ok=True)
PAIRS = {  # pair: (target, draft, c = draft/target params)
    "Q1": ("qwen17", "qwen06", 0.6/1.7), "Q2": ("qwen4", "qwen06", 0.6/4), "Q4": ("qwen4", "qwen17", 1.7/4),
    "Q5": ("qwen8", "qwen06", 0.6/8), "Q6": ("qwen8", "qwen17", 1.7/8), "Q3": ("qwen8", "qwen4", 4/8),
    "G2": ("gemma12", "gemma1", 1/12), "G1": ("gemma12", "gemma4", 4/12), "M1": ("ministral8", "ministral3", 3/8),
}
COLS = ["doc_id", "pool_index", "position", "target_token_id", "argmax_id", "logp_argmax", "entropy", "top2_margin"]

def tf_dir(kind, target, model, base=ROOT):
    d = os.path.join(base, "teacher_forced", kind, target, model)
    sub = sorted(os.listdir(d))
    return os.path.join(d, sub[0])

def read_tf(kind, target, model, cols=COLS, common_only=True, base=ROOT):
    files = sorted(glob.glob(os.path.join(tf_dir(kind, target, model, base), "shard-*.parquet")))
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    if common_only and "pool_index" in df:
        df = df[df.pool_index < 20000]
    return df.sort_values(["pool_index", "position"], kind="stable").reset_index(drop=True)

def load_pair(kind, target, draft, common_only=True, base=ROOT):
    t = read_tf(kind, target, target, common_only=common_only, base=base)
    d = read_tf(kind, target, draft, common_only=common_only, base=base)
    assert len(t) == len(d) and (t.doc_id.values == d.doc_id.values).all() and (t.position.values == d.position.values).all()
    out = pd.DataFrame({
        "doc": t.pool_index.values.astype(np.int64), "pos": t.position.values.astype(np.int32),
        "y": t.target_token_id.values, "t_arg": t.argmax_id.values,
        "mT": t.top2_margin.values, "HT": t.entropy.values, "t_lp1": t.logp_argmax.values,
        "d_arg": d.argmax_id.values, "mD": d.top2_margin.values, "HD": d.entropy.values, "d_lp1": d.logp_argmax.values,
    })
    return out

ACOLS = ["doc_id", "pool_index", "generated_token_position", "token_start", "token_end", "morpheme_char_spans",
         "alignment_class", "alignment_status", "is_first_token"]

def read_aligned(target, split, base=ROOT, common_only=True):
    d = os.path.join(base, "aligned", "kiwi", target, split)
    files = sorted(glob.glob(os.path.join(d, "**", "shard-*.parquet"), recursive=True))
    a = pd.concat([pd.read_parquet(f, columns=ACOLS) for f in files], ignore_index=True)
    if common_only: a = a[a.pool_index < 20000]
    ok = (a.alignment_status.values == "ALIGNED")
    st = a.token_start.values; en = a.token_end.values
    spans = a.morpheme_char_spans.values
    ptype = np.full(len(a), "excluded", dtype=object)
    first = a.is_first_token.values
    for i in np.flatnonzero(ok):
        if first[i] or st[i] == 0:
            ptype[i] = "word_onset"; continue
        b = [int(s[0]) for s in spans[i][1:]]
        if st[i] in b: ptype[i] = "morph_onset"
        elif any(st[i] < x < en[i] for x in b): ptype[i] = "fused"
        else: ptype[i] = "internal"
    out = pd.DataFrame({"doc": a.pool_index.values.astype(np.int64), "pos": a.generated_token_position.values.astype(np.int32),
                        "ptype": ptype, "aclass": a.alignment_class.values})
    return out

def std_ratio(df, num, den, by, nbins=10, col="ptype", y="rej"):
    """Direct standardization: rejection rates within quantile bins of `by`, weighted by all aligned positions."""
    d = df[df[col] != "excluded"] if col == "ptype" else df[df[col].isin(["CROSS", "WITHIN_SPLIT", "EXACT"])]
    cols = by if isinstance(by, list) else [by]
    key = np.zeros(len(d), np.int64)
    for cname in cols:
        q = pd.qcut(d[cname].rank(method="first"), nbins, labels=False).values
        key = key * nbins + q
    d = d.assign(_b=key)
    w = d.groupby("_b").size()
    def rate(v):
        g = d[d[col] == v].groupby("_b")[y].mean()
        common = g.index
        return g, common
    g1, c1 = rate(num); g2, c2 = rate(den)
    common = c1.intersection(c2)
    ww = w.loc[common] / w.loc[common].sum()
    return float((g1.loc[common] * ww).sum() / (g2.loc[common] * ww).sum()), float(d[d[col] == num][y].mean() / d[d[col] == den][y].mean())
