"""E-A: adaptive-drafting baselines, oracle, held-out thresholds, document bootstrap."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from ksdlib import *; from policysim import *
from sklearn.ensemble import HistGradientBoostingClassifier

PAIRS_ALL = dict(PAIRS)
O1 = ROOT + "/o1_qwen14"
EXTRA = {"Q7": ("qwen14", "qwen06", 0.6/14), "Q8": ("qwen14", "qwen17", 1.7/14), "Q9": ("qwen14", "qwen4", 4/14)}
rng = np.random.default_rng(20261125)
B = 500
rows = []; perdoc_rows = []
for pid, (tgt, drf, c) in list(PAIRS_ALL.items()) + list(EXTRA.items()):
    t0 = time.time()
    base = O1 if pid in EXTRA else ROOT
    df = load_pair("generated", tgt, drf, base=base)
    starts = doc_starts(df.doc.values); n = np.diff(starts).astype(float); nd = len(n)
    docid = df.doc.values[starts[:-1]]
    match = (df.d_arg.values == df.y.values)
    dev_doc = (docid % 2 == 0); test_doc = ~dev_doc
    never = np.zeros(len(df), bool)
    # learned stopper (scalar draft features), trained on dev documents only
    tok_dev = np.repeat(dev_doc, np.diff(starts))
    X = np.c_[df.mD.values, df.HD.values, df.d_lp1.values]
    idx = np.flatnonzero(tok_dev); idx = rng.choice(idx, min(400_000, len(idx)), replace=False)
    clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, max_leaf_nodes=31, random_state=0)
    clf.fit(X[idx], ~match[idx])
    p_rej = clf.predict_proba(X)[:, 1]
    signals = {  # name: (signal, stop when signal < thr ?)
        "margin (ours)": (df.mD.values, True),
        "entropy (AdaEDL/SVIP-type)": (-df.HD.values, True),
        "max-prob (confidence thr.)": (df.d_lp1.values, True),
        "learned stopper (GBDT)": (-p_rej, True),
    }
    def run_fixed(k):
        cc, dd, _ = simulate(match, never, starts, k); return cc.astype(float), dd.astype(float)
    fixed = {k: run_fixed(k) for k in range(1, 9)}
    grids = {}
    for name, (sig, _) in signals.items():
        u = np.unique(sig)
        thr = u if len(u) <= 60 else np.unique(np.quantile(sig, np.linspace(0.01, 0.75, 60)))
        C = np.zeros((len(thr), nd)); D = np.zeros((len(thr), nd))
        for i, th in enumerate(thr):
            cc, dd, _ = simulate(match, sig < th, starts, 16); C[i] = cc; D[i] = dd
        grids[name] = (thr, C, D)
    thr = -np.r_[np.linspace(0.05, 3, 30), np.linspace(3.2, 12, 20)]
    C = np.zeros((len(thr), nd)); D = np.zeros((len(thr), nd))
    for i, th in enumerate(thr):
        cc, dd = simulate_cum(match, df.d_lp1.values, starts, 16, th); C[i] = cc; D[i] = dd
    grids["cumulative log-prob"] = (thr, C, D)
    oc, od = simulate_oracle(match, starts, 16); oc = oc.astype(float); od = od.astype(float)

    def LD(Cw, Dw, nw): return nw / Cw, Dw / Cw
    def at_cost(L, D, D0):
        o = np.argsort(D); return float(np.interp(D0, D[o], L[o]))
    def evaluate(w):
        """w: document weights (test docs only). Returns dict of metrics."""
        nw = (n * w).sum(); out = {}
        c4, d4 = fixed[4]; L4, D4 = LD((c4 * w).sum(), (d4 * w).sum(), nw)
        out["L_k4"] = L4; out["D_k4"] = D4
        fx = [LD((fixed[k][0] * w).sum(), (fixed[k][1] * w).sum(), nw) for k in range(1, 9)]
        out["S_fixed"] = max(L / (1 + c * D) for L, D in fx)
        for name, (thr, C, D) in grids.items():
            L, Dd = LD(C @ w, D @ w, nw)
            out["L@k4 " + name] = at_cost(L, Dd, D4)
            out["S " + name] = float((L / (1 + c * Dd)).max())
        Lo, Do = LD((oc * w).sum(), (od * w).sum(), nw)
        out["L oracle"] = Lo; out["D oracle"] = Do; out["S oracle"] = Lo / (1 + c * Do)
        return out
    w_test = test_doc.astype(float)
    point = evaluate(w_test)
    # held-out: thresholds chosen on dev docs for best est. speedup, applied to test docs
    w_dev = dev_doc.astype(float)
    held = {}
    for name, (thr, C, D) in grids.items():
        L, Dd = LD(C @ w_dev, D @ w_dev, (n * w_dev).sum()); i = int(np.argmax(L / (1 + c * Dd)))
        Lt, Dt = LD((C[i] * w_test).sum(), (D[i] * w_test).sum(), (n * w_test).sum())
        held[name] = Lt / (1 + c * Dt)
    kbest = max(range(1, 9), key=lambda k: LD((fixed[k][0] * w_dev).sum(), (fixed[k][1] * w_dev).sum(), (n * w_dev).sum())[0] / (1 + c * (fixed[k][1] * w_dev).sum() / (fixed[k][0] * w_dev).sum()))
    Lt, Dt = LD((fixed[kbest][0] * w_test).sum(), (fixed[kbest][1] * w_test).sum(), (n * w_test).sum())
    held["fixed"] = Lt / (1 + c * Dt)
    # document bootstrap on test docs
    ti = np.flatnonzero(test_doc); boots = []
    for b in range(B):
        w = np.zeros(nd); np.add.at(w, rng.choice(ti, len(ti), replace=True), 1.0)
        boots.append(evaluate(w))
    bdf = pd.DataFrame(boots)
    def ci(series): return [float(np.quantile(series, 0.025)), float(np.quantile(series, 0.975))]
    rec = {"pair": pid, "c": round(c, 3), "docs_test": int(test_doc.sum()), **{k: round(v, 4) for k, v in point.items()}}
    for name in grids:
        g = 100 * (bdf["L@k4 " + name] / bdf["L_k4"] - 1)
        rec["gain% " + name] = round(100 * (point["L@k4 " + name] / point["L_k4"] - 1), 2); rec["gain_ci " + name] = [round(x, 2) for x in ci(g)]
        rec["heldout_S " + name] = round(held[name], 4)
    for other in ["entropy (AdaEDL/SVIP-type)", "max-prob (confidence thr.)", "cumulative log-prob", "learned stopper (GBDT)"]:
        g = 100 * (bdf["L@k4 margin (ours)"] / bdf["L@k4 " + other] - 1)
        rec["margin_vs " + other] = [round(100 * (point["L@k4 margin (ours)"] / point["L@k4 " + other] - 1), 2)] + [round(x, 2) for x in ci(g)]
    rec["heldout_S fixed"] = round(held["fixed"], 4)
    rec["S_margin_vs_fixed_ci"] = [round(x, 2) for x in ci(100 * (bdf["S margin (ours)"] / bdf["S_fixed"] - 1))]
    rec["oracle_gain%"] = round(100 * (point["L oracle"] / point["L_k4"] - 1), 1)
    # per-document variability: threshold nearest to k=4 cost on dev, per-doc ratio of calls on test
    thr, C, D = grids["margin (ours)"]
    L, Dd = LD(C @ w_dev, D @ w_dev, (n * w_dev).sum()); D4dev = (fixed[4][1] * w_dev).sum() / (fixed[4][0] * w_dev).sum()
    i = int(np.argmin(np.abs(Dd - D4dev)))
    long = test_doc & (n >= 64)
    per = 100 * (fixed[4][0][long] / C[i][long] - 1)  # per-document gain in tokens per call
    rec["perdoc_thr"] = float(thr[i]); rec["perdoc_cost_ratio"] = round(float((D[i][long].sum() / C[i][long].sum()) / (fixed[4][1][long].sum() / fixed[4][0][long].sum())), 3)
    rec["perdoc_gain_median"] = round(float(np.median(per)), 1); rec["perdoc_gain_iqr"] = [round(float(np.quantile(per, .25)), 1), round(float(np.quantile(per, .75)), 1)]
    rec["perdoc_share_positive"] = round(float((per > 0).mean()), 3); rec["perdoc_share_nonneg"] = round(float((per >= 0).mean()), 3)
    rows.append(rec)
    print(pid, round(time.time() - t0, 1), "s", {k: rec[k] for k in ["L_k4", "L@k4 margin (ours)", "L@k4 entropy (AdaEDL/SVIP-type)", "L@k4 max-prob (confidence thr.)", "L@k4 cumulative log-prob", "L@k4 learned stopper (GBDT)", "L oracle"]}, flush=True)
json.dump(rows, open(OUT / "expA_policies.json", "w"), indent=1)
