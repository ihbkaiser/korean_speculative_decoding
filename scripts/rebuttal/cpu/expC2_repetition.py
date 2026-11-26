"""E-C2: do the main results survive when repetitive (looping) text is removed?"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ksdlib import *; from policysim import doc_starts
from sklearn.metrics import roc_auc_score
from expC_helpers import verified_k4
out = []
for pid in ["Q2", "Q5", "M1", "G1"]:
    tgt, drf, _ = PAIRS[pid]
    df = load_pair("generated", tgt, drf)
    st = doc_starts(df.doc.values); y = df.y.values
    rep = np.zeros(len(df), bool)   # token completes a 4-gram already seen earlier in this continuation
    for d in range(len(st) - 1):
        s, e = st[d], st[d + 1]; seen = set()
        for i in range(s + 3, e):
            g = (y[i - 3], y[i - 2], y[i - 1], y[i])
            if g in seen: rep[i] = True
            seen.add(g)
    df["rep"] = rep
    al = read_aligned(tgt, "all" if tgt == "qwen4" else "common")
    df = df.merge(al, on=["doc", "pos"], how="left"); df["ptype"] = df.ptype.fillna("excluded"); df["aclass"] = df.aclass.fillna("OTHER")
    df = df[verified_k4(df)].copy(); df["rej"] = (df.d_arg != df.y).astype(int)
    for name, s in [("all verified", df), ("non-repeated only", df[~df.rep]), ("repeated only", df[df.rep])]:
        lo = s.mT <= s.mT.quantile(0.2)
        rec = {"pair": pid, "subset": name, "share of proposals%": round(100 * len(s) / len(df), 1), "rej%": round(100 * s.rej.mean(), 2),
               "DP pos%": round(100 * (s.mT < 1).mean(), 1), "DP rej_share%": round(100 * s.rej[s.mT < 1].sum() / s.rej.sum(), 1),
               "low20 rej_share%": round(100 * s.rej[lo].sum() / s.rej.sum(), 1), "AUC mT": round(roc_auc_score(s.rej, -s.mT), 3), "AUC mD": round(roc_auc_score(s.rej, -s.mD), 3)}
        x = s[s.ptype != "excluded"]; rec["AUC boundary"] = round(roc_auc_score(x.rej, (x.ptype != "internal").astype(int)), 3)
        by = ["mT", "mD"] if pid.startswith("G") else "mT"
        for nm, (a, b, col) in {"morph/int": ("morph_onset", "internal", "ptype"), "word/int": ("word_onset", "internal", "ptype"), "fused/split": ("CROSS", "WITHIN_SPLIT", "aclass")}.items():
            try:
                m, raw = std_ratio(s, a, b, by, col=col); rec[nm] = [round(raw, 2), round(m, 2)]
            except Exception as ex: rec[nm] = str(ex)[:40]
        out.append(rec); print(json.dumps(rec), flush=True)
json.dump(out, open(OUT / "expC2_repetition.json", "w"), indent=1)
