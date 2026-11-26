"""E-C: the confident-target regime (Gemma) and how it connects to the other pairs (generated text only)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ksdlib import *; from policysim import doc_starts
from sklearn.metrics import roc_auc_score
O1 = ROOT + "/o1_qwen14"
EXTRA = {"Q7": ("qwen14", "qwen06", 0), "Q8": ("qwen14", "qwen17", 0), "Q9": ("qwen14", "qwen4", 0)}
res = {"fork2x2": [], "repetition": [], "strata": []}
def verified_k4(df):
    """mark positions verified under k=4 greedy SD (replay)"""
    from policysim import simulate
    starts = doc_starts(df.doc.values); match = (df.d_arg.values == df.y.values)
    ver = np.zeros(len(df), bool)
    for d in range(len(starts) - 1):
        c, e = starts[d], starts[d + 1]
        while c < e:
            j = 0; mis = -1
            while c + j < e and j < 4:
                ver[c + j] = True
                if not match[c + j]: mis = c + j; break
                j += 1
            c = mis + 1 if mis >= 0 else c + j + 1
    return ver
seen_t = set()
for pid, (tgt, drf, _) in list(PAIRS.items()) + list(EXTRA.items()):
    base = O1 if pid in EXTRA else ROOT
    df = load_pair("generated", tgt, drf, base=base)
    df = df[verified_k4(df)].copy(); df["rej"] = (df.d_arg != df.y).astype(int)
    tdp = df.mT < 1; ddp = df.mD < 1; R = df.rej.sum()
    cells = {"both": tdp & ddp, "target only": tdp & ~ddp, "draft only": ~tdp & ddp, "neither": ~tdp & ~ddp}
    rec = {"pair": pid, "rej%": round(100 * df.rej.mean(), 2)}
    for k, m in cells.items():
        rec[k] = {"pos%": round(100 * m.mean(), 1), "rej_share%": round(100 * df.rej[m].sum() / R, 1), "rej_rate%": round(100 * df.rej[m].mean(), 1)}
    rec["either-fork rej_share%"] = round(100 * df.rej[tdp | ddp].sum() / R, 1); rec["either-fork pos%"] = round(100 * (tdp | ddp).mean(), 1)
    res["fork2x2"].append(rec); print(json.dumps(rec), flush=True)
    if tgt not in seen_t:
        seen_t.add(tgt)
        full = read_tf("generated", tgt, tgt, cols=["pool_index", "position", "target_token_id", "top2_margin"], base=base)
        st = doc_starts(full.pool_index.values); y = full.target_token_id.values
        rep4 = 0; tot = 0; uniq = []
        for d in range(len(st) - 1):
            seq = y[st[d]:st[d + 1]]
            if len(seq) < 32: continue
            seen = set(); r = 0
            for i in range(len(seq) - 3):
                g = (seq[i], seq[i + 1], seq[i + 2], seq[i + 3])
                if g in seen: r += 1
                seen.add(g)
            rep4 += r; tot += len(seq) - 3; uniq.append(len(set(seq.tolist())) / len(seq))
        rr = {"target": tgt, "repeated 4-gram rate%": round(100 * rep4 / tot, 1), "median distinct-token ratio": round(float(np.median(uniq)), 3),
              "median margin": round(float(np.median(full.top2_margin)), 2), "DP share%": round(100 * float((full.top2_margin < 1).mean()), 1)}
        res["repetition"].append(rr); print(json.dumps(rr), flush=True)
    if pid in ("Q2", "M1"):
        # stratify documents by target confidence (mean margin) -> does a confident stratum look like Gemma?
        g = df.groupby("doc").mT.mean(); q = pd.qcut(g, 10, labels=False); df["dec"] = df.doc.map(q)
        for dec in [0, 4, 9]:
            s = df[df.dec == dec]; lo = s.mT <= s.mT.quantile(0.2)
            rr = {"pair": pid, "doc decile by mean target margin": dec + 1, "rej%": round(100 * s.rej.mean(), 2), "DP pos%": round(100 * (s.mT < 1).mean(), 1),
                  "DP rej_share%": round(100 * s.rej[s.mT < 1].sum() / s.rej.sum(), 1), "low20 rej_share%": round(100 * s.rej[lo].sum() / s.rej.sum(), 1),
                  "AUC mT": round(roc_auc_score(s.rej, -s.mT), 3), "AUC mD": round(roc_auc_score(s.rej, -s.mD), 3)}
            res["strata"].append(rr); print(json.dumps(rr), flush=True)
json.dump(res, open(OUT / "expC_gemma.json", "w"), indent=1)
