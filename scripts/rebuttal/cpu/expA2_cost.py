"""E-A2: (i) how the margin policy's advantage depends on the draft/target cost ratio c;
(ii) fixed-k grid extended to 16; (iii) one universal threshold (1 nat) for every pair."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ksdlib import *; from policysim import *
O1 = ROOT + "/o1_qwen14"
ALL = dict(PAIRS); ALL.update({"Q7": ("qwen14", "qwen06", 0.6/14), "Q8": ("qwen14", "qwen17", 1.7/14), "Q9": ("qwen14", "qwen4", 4/14)})
CS = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
out = []
for pid, (tgt, drf, c0) in ALL.items():
    df = load_pair("generated", tgt, drf, base=O1 if pid in ("Q7", "Q8", "Q9") else ROOT)
    starts = doc_starts(df.doc.values); N = float(np.diff(starts).sum()); match = (df.d_arg.values == df.y.values); never = np.zeros(len(df), bool)
    def LD(stop, kmax):
        c, d, _ = simulate(match, stop, starts, kmax); return N / c.sum(), d.sum() / c.sum()
    fixed = {k: LD(never, k) for k in range(1, 17)}
    pol = {}
    for name, sig in {"margin": df.mD.values, "entropy": -df.HD.values, "maxprob": df.d_lp1.values}.items():
        u = np.unique(sig); thr = u if len(u) <= 80 else np.unique(np.quantile(sig, np.linspace(0.005, 0.9, 80)))
        pol[name] = [LD(sig < t, 16) for t in thr]
    L1, D1 = LD(df.mD.values < 1.0, 16)
    rec = {"pair": pid, "c_param": round(c0, 3), "tau=1nat": {"L": round(L1, 3), "D": round(D1, 3)}, "by_c": {}}
    for c in CS + [c0]:
        best_fixed8 = max(L / (1 + c * D) for k, (L, D) in fixed.items() if k <= 8)
        kbest, best_fixed16 = max(((k, L / (1 + c * D)) for k, (L, D) in fixed.items()), key=lambda x: x[1])
        r = {"fixed<=8": round(best_fixed8, 3), "fixed<=16": round(best_fixed16, 3), "k*": kbest}
        for name, lst in pol.items(): r[name] = round(max(L / (1 + c * D) for L, D in lst), 3)
        r["tau=1"] = round(L1 / (1 + c * D1), 3)
        r["margin_vs_fixed16%"] = round(100 * (r["margin"] / best_fixed16 - 1), 1); r["tau1_vs_fixed16%"] = round(100 * (r["tau=1"] / best_fixed16 - 1), 1)
        r["margin_vs_entropy%"] = round(100 * (r["margin"] / r["entropy"] - 1), 1); r["margin_vs_maxprob%"] = round(100 * (r["margin"] / r["maxprob"] - 1), 1)
        rec["by_c"]["param" if c == c0 and c not in CS else str(c)] = r
    out.append(rec)
    print(pid, "tau=1:", rec["tau=1nat"], "| param c:", rec["by_c"].get("param", rec["by_c"].get(str(c0))), flush=True)
    print("    gain of margin over best fixed k<=16 by c:", {c: rec["by_c"][str(c)]["margin_vs_fixed16%"] for c in CS}, flush=True)
    print("    est. speedup margin by c:", {c: rec["by_c"][str(c)]["margin"] for c in CS}, flush=True)
json.dump(out, open(OUT / "expA2_cost.json", "w"), indent=1)
