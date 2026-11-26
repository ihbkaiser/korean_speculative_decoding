"""E-D2: inside non-Gemma pairs, do the most confident documents behave like Gemma?
Per document-confidence stratum (deciles of the document's mean target margin): rejection rate,
where rejections sit (target fork / draft-only fork / neither), and position-type ratios matched on
the target margin alone vs on both margins. Document bootstrap intervals for every stratum."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ksdlib import *
from expC_helpers import verified_k4
from expD_lib import ratio, CONTRASTS, keys_for, boot_ci

rng = np.random.default_rng(20261125)
out = []
for pid in ["Q2", "Q5", "Q3", "M1", "G1", "G2"]:
    tgt, drf, _ = PAIRS[pid]
    df = load_pair("generated", tgt, drf)
    al = read_aligned(tgt, "all" if tgt == "qwen4" else "common")
    df = df.merge(al, on=["doc", "pos"], how="left"); df["ptype"] = df.ptype.fillna("excluded"); df["aclass"] = df.aclass.fillna("OTHER")
    df = df[verified_k4(df)].reset_index(drop=True); df["rej"] = (df.d_arg != df.y).astype(int)
    g = df.groupby("doc").mT.mean(); dec = pd.qcut(g, 10, labels=False); df["dec"] = df.doc.map(dec).values
    strata = {"all documents": df.index.values} if pid.startswith("G") else {
        "least confident 10%": df.index[df.dec == 0].values, "middle 20%": df.index[df.dec.isin([4, 5])].values, "most confident 10%": df.index[df.dec == 9].values}
    for name, idx in strata.items():
        s = df.loc[idx].reset_index(drop=True); R = s.rej.sum(); tdp = s.mT < 1; ddp = s.mD < 1
        rec = {"pair": pid, "stratum": name, "proposals": int(len(s)), "rejections": int(R), "rej%": round(100 * s.rej.mean(), 2),
               "median target margin": round(float(s.mT.median()), 2), "target DP pos%": round(100 * float(tdp.mean()), 1),
               "rej share: target fork%": round(100 * float(s.rej[tdp].sum() / R), 1), "rej share: draft-only fork%": round(100 * float(s.rej[~tdp & ddp].sum() / R), 1),
               "rej share: neither%": round(100 * float(s.rej[~tdp & ~ddp].sum() / R), 1)}
        K = keys_for(s)
        for kn in ["raw", "target abs bins", "both abs bins"]:
            rec[kn] = {cn: round(ratio(s, a, b, col, K[kn])[0], 2) for cn, (a, b, col) in CONTRASTS.items()}
        # document bootstrap (500 resamples) on per-document cell counts; absolute bins do not move under resampling
        rec["ci95"] = {kn: {cn: boot_ci(s, a, b, col, K[kn], rng) for cn, (a, b, col) in CONTRASTS.items()} for kn in ["target abs bins", "both abs bins"]}
        out.append(rec); print(json.dumps(rec), flush=True)
json.dump(out, open(OUT / "expD2_confident_strata.json", "w"), indent=1)
