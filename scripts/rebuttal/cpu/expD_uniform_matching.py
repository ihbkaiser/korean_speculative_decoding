"""E-D: one matching strategy for every pair (answers z5Go W3).
Does Gemma need joint binning because its mechanism differs, or because decile bins of the target
margin put almost no resolution where its rejections are? Compare, for all pairs, position-type
ratios matched on: target-margin deciles, target-margin absolute bins (nats), draft-margin deciles,
both margins (deciles), both margins (absolute bins)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ksdlib import *
from expC_helpers import verified_k4

O1 = ROOT + "/o1_qwen14"
ALL = dict(PAIRS); ALL.update({"Q7": ("qwen14", "qwen06", 0), "Q8": ("qwen14", "qwen17", 0), "Q9": ("qwen14", "qwen4", 0)})
EDGES = np.array([0.125, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0])   # nats; 11 bins
CONTRASTS = {"morph/int": ("morph_onset", "internal", "ptype"), "word/int": ("word_onset", "internal", "ptype"), "fused/split": ("CROSS", "WITHIN_SPLIT", "aclass")}

def ratio(d, num, den, col, key):
    """direct standardization over cells of `key`, weights = all aligned positions in cells where both classes occur"""
    d = d[d[col] != "excluded"] if col == "ptype" else d[d[col].isin(["CROSS", "WITHIN_SPLIT", "EXACT"])]
    k = key[d.index.values] if key is not None else np.zeros(len(d), np.int64)
    g = pd.DataFrame({"k": k, "rej": d.rej.values, "c": d[col].values})
    w = g.groupby("k").size()
    a = g[g.c == num].groupby("k").rej.mean(); b = g[g.c == den].groupby("k").rej.mean()
    common = a.index.intersection(b.index); ww = w.loc[common] / w.loc[common].sum()
    return float((a.loc[common] * ww).sum() / (b.loc[common] * ww).sum()), float(w.loc[common].sum() / w.sum())

out = []
for pid, (tgt, drf, _) in ALL.items():
    base = O1 if pid in ("Q7", "Q8", "Q9") else ROOT
    df = load_pair("generated", tgt, drf, base=base)
    al = read_aligned(tgt, "all" if tgt == "qwen4" else "common", base=base)
    df = df.merge(al, on=["doc", "pos"], how="left"); df["ptype"] = df.ptype.fillna("excluded"); df["aclass"] = df.aclass.fillna("OTHER")
    df = df[verified_k4(df)].reset_index(drop=True); df["rej"] = (df.d_arg != df.y).astype(int)
    qT = pd.qcut(df.mT.rank(method="first"), 10, labels=False).values; qD = pd.qcut(df.mD.rank(method="first"), 10, labels=False).values
    aT = np.digitize(df.mT.values, EDGES); aD = np.digitize(df.mD.values, EDGES)
    keys = {"raw": None, "target deciles": qT, "target abs bins": aT, "draft deciles": qD, "draft abs bins": aD,
            "both deciles": qT * 10 + qD, "both abs bins": aT * 11 + aD}
    rec = {"pair": pid, "rej%": round(100 * df.rej.mean(), 2),
           "share of positions in lowest target decile with margin<1": round(float((df.mT[qT == 0] < 1).mean()), 3),
           "share of rejections in lowest target decile": round(float(df.rej[qT == 0].sum() / df.rej.sum()), 3),
           "target-margin range of lowest decile (nats)": [round(float(df.mT[qT == 0].min()), 2), round(float(df.mT[qT == 0].max()), 2)]}
    for name, key in keys.items():
        rec[name] = {}
        for cn, (a, b, col) in CONTRASTS.items():
            r, cov = ratio(df, a, b, col, key); rec[name][cn] = round(r, 2)
            if name != "raw": rec[name][cn + " coverage"] = round(cov, 2)
    out.append(rec)
    print(pid, rec["rej%"], "| lowest decile:", rec["share of rejections in lowest target decile"], rec["target-margin range of lowest decile (nats)"], flush=True)
    for name in keys: print("    %-16s" % name, {k: v for k, v in rec[name].items() if "coverage" not in k}, flush=True)
json.dump(out, open(OUT / "expD_uniform_matching.json", "w"), indent=1)
