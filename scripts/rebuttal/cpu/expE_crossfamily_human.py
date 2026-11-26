"""E-E: outside-model margin from a DIFFERENT family, human-written text, CPU only.
Both families scored the same documents from the document start, each with its own tokenizer.
Positions are joined on the document-level character offset at which a token starts
(`global_token_char_span[0]` in the alignment tables), so both models have read exactly the same prefix."""
import os, sys, json, glob
sys.path.insert(0, os.environ.get("KSD_CPU", os.path.dirname(os.path.abspath(__file__))))
from ksdlib import *
from sklearn.metrics import roc_auc_score
from expD_lib import EDGES, ratio as cell_ratio, boot_ci

BOOTSTRAPS = 500

def bootstrap_auc_ci(frame, labels, scores, rng, reps=BOOTSTRAPS):
    """Document bootstrap CI for AUC, using fixed score ordering and resample weights."""
    y = np.asarray(labels, dtype=np.int8)
    score = np.asarray(scores, dtype=np.float64)
    docs, doc_code = np.unique(frame.doc.values, return_inverse=True)
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    starts = np.r_[0, np.flatnonzero(sorted_score[1:] != sorted_score[:-1]) + 1]
    group = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(order)]))
    sorted_doc = doc_code[order]
    sorted_y = y[order]
    n_groups = len(starts)
    estimates = np.empty(reps, dtype=np.float64)
    for i in range(reps):
        doc_weights = np.bincount(rng.integers(0, len(docs), len(docs)), minlength=len(docs))
        weights = doc_weights[sorted_doc]
        pos = np.bincount(group, weights=weights * sorted_y, minlength=n_groups)
        neg = np.bincount(group, weights=weights * (1 - sorted_y), minlength=n_groups)
        neg_before = np.cumsum(neg) - neg
        denom = pos.sum() * neg.sum()
        estimates[i] = np.sum(pos * (neg_before + 0.5 * neg)) / denom if denom else np.nan
    return [round(float(np.nanquantile(estimates, 0.025)), 3), round(float(np.nanquantile(estimates, 0.975)), 3)]

def margin_decile_keys(frame, margin, contrast_col):
    """Return full-frame keys equivalent to std_ratio's per-contrast decile bins."""
    if contrast_col == "ptype":
        mask = frame.ptype.to_numpy() != "excluded"
    else:
        mask = frame[contrast_col].isin(["CROSS", "WITHIN_SPLIT", "EXACT"]).to_numpy()
    key = np.full(len(frame), -1, dtype=np.int64)
    key[np.flatnonzero(mask)] = pd.qcut(
        frame.loc[mask, margin].rank(method="first"), 10, labels=False
    ).to_numpy(dtype=np.int64)
    return key

def aligned_with_offsets(target):
    a = read_aligned(target, "human")                       # doc, pos, ptype, aclass (same row order as files)
    d = os.path.join(ROOT, "aligned", "kiwi", target, "human")
    files = sorted(glob.glob(os.path.join(d, "**", "shard-*.parquet"), recursive=True))
    g = pd.concat([pd.read_parquet(f, columns=["pool_index", "generated_token_position", "global_token_char_span"]) for f in files], ignore_index=True)
    g = g[g.pool_index < 20000]
    off = pd.DataFrame({"doc": g.pool_index.values.astype(np.int64), "pos": g.generated_token_position.values.astype(np.int32),
                        "cstart": [int(s[0]) for s in g.global_token_char_span.values]})
    return a.merge(off, on=["doc", "pos"], how="left")

def tf(host, model, cols=("top2_margin", "argmax_id", "entropy")):
    t = read_tf("human", host, model)
    out = pd.DataFrame({"doc": t.pool_index.values.astype(np.int64), "pos": t.position.values.astype(np.int32)})
    for c in cols: out[c] = t[c].values
    return out

QH, MH = "qwen8", "ministral8"       # host directories: all Qwen3 models scored qwen8's human refs; Ministral models scored ministral8's
qa, ma = aligned_with_offsets(QH), aligned_with_offsets(MH)
qm = {m: tf(QH, m) for m in ["qwen06", "qwen17", "qwen4", "qwen8"]}
mm = {m: tf(MH, m) for m in ["ministral3", "ministral8"]}

def build(pair_side, draft, target, own_judges, other_side_judges):
    A, own, other, oa = (qa, qm, mm, ma) if pair_side == "qwen" else (ma, mm, qm, qa)
    d = A.merge(own[target].rename(columns={"top2_margin": "mT", "argmax_id": "t_arg", "entropy": "HT"}), on=["doc", "pos"])
    d = d.merge(own[draft][["doc", "pos", "argmax_id"]].rename(columns={"argmax_id": "d_arg"}), on=["doc", "pos"])
    d["rej"] = (d.t_arg != d.d_arg).astype(int)
    for j in own_judges:
        d = d.merge(own[j][["doc", "pos", "top2_margin"]].rename(columns={"top2_margin": "m_" + j}), on=["doc", "pos"])
    full_n = len(d)
    o = oa[["doc", "pos", "cstart"]].rename(columns={"pos": "opos"})
    for j in other_side_judges:
        o = o.merge(other[j][["doc", "pos", "top2_margin"]].rename(columns={"pos": "opos", "top2_margin": "m_" + j}), on=["doc", "opos"])
    o = o.drop_duplicates(["doc", "cstart"])
    s = d.merge(o, on=["doc", "cstart"], how="inner").reset_index(drop=True)
    return d, s, full_n

def report(tag, d, s, judges_all):
    x = s[s.ptype != "excluded"]
    rec = {"setting": tag, "positions_all": int(len(d)), "positions_shared_offset": int(len(s)), "coverage": round(len(s) / len(d), 3),
           "coverage_by_type": {k: round(float(v), 3) for k, v in (s.groupby("ptype").size() / d.groupby("ptype").size()).items()},
           "disagreement_all": round(float(d.rej.mean()), 4), "disagreement_shared": round(float(s.rej.mean()), 4),
           "AUC boundary label": round(roc_auc_score(x.rej, (x.ptype != "internal").astype(int)), 3),
           "AUC target margin": round(roc_auc_score(s.rej, -s.mT), 3)}
    rng = np.random.default_rng(20261125)
    auc_ci = {
        "boundary label": bootstrap_auc_ci(x, x.rej, (x.ptype != "internal").astype(int), rng),
        "target margin": bootstrap_auc_ci(s, s.rej, -s.mT, rng),
    }
    def ratios(by):
        r = {}
        for nm, (a, b, col) in {"morph/int": ("morph_onset", "internal", "ptype"), "word/int": ("word_onset", "internal", "ptype"), "fused/split": ("CROSS", "WITHIN_SPLIT", "aclass")}.items():
            m, raw = std_ratio(s, a, b, by, col=col); r[nm] = [round(raw, 2), round(m, 2)]
        return r
    rec["matched on target margin"] = ratios("mT"); rec["matched on target entropy"] = ratios("HT")
    for j in judges_all:
        rec["AUC judge " + j] = round(roc_auc_score(s.rej, -s["m_" + j]), 3)
        auc_ci["judge " + j] = bootstrap_auc_ci(s, s.rej, -s["m_" + j], rng)
        rec["spearman(target margin, judge " + j + ")"] = round(float(s[["mT", "m_" + j]].corr(method="spearman").iloc[0, 1]), 3)
        rec["matched on judge " + j] = ratios("m_" + j)
        judge_abs = np.digitize(s["m_" + j].values, EDGES)
        rec["matched on judge " + j + " absolute bins"] = {}
        rec["matched judge ratio 95% CI: " + j] = {}
        for nm, (a, b, col) in {"morph/int": ("morph_onset", "internal", "ptype"), "word/int": ("word_onset", "internal", "ptype"), "fused/split": ("CROSS", "WITHIN_SPLIT", "aclass")}.items():
            decile_key = margin_decile_keys(s, "m_" + j, col)
            absolute_ratio = cell_ratio(s, a, b, col, judge_abs)[0]
            rec["matched on judge " + j + " absolute bins"][nm] = [
                rec["matched on judge " + j][nm][0], round(absolute_ratio, 2)
            ]
            rec["matched judge ratio 95% CI: " + j][nm] = {
                "deciles": boot_ci(s, a, b, col, decile_key, rng, reps=BOOTSTRAPS),
                "absolute bins": boot_ci(s, a, b, col, judge_abs, rng, reps=BOOTSTRAPS),
            }
    rec["AUC 95% CI"] = auc_ci
    print(json.dumps(rec, ensure_ascii=False), flush=True); return rec

out = []
for draft, target, own in [("qwen06", "qwen4", ["qwen8"]), ("qwen06", "qwen8", ["qwen4"]), ("qwen17", "qwen4", ["qwen8"]), ("qwen4", "qwen8", ["qwen17"])]:
    d, s, n = build("qwen", draft, target, own, ["ministral8", "ministral3"])
    out.append(report(f"human {draft}->{target}; cross-family judges Ministral", d, s, own + ["ministral8", "ministral3"]))
d, s, n = build("ministral", "ministral3", "ministral8", [], ["qwen8", "qwen4", "qwen17", "qwen06"])
out.append(report("human ministral3->ministral8; cross-family judges Qwen3", d, s, ["qwen8", "qwen4", "qwen17", "qwen06"]))
json.dump(out, open(OUT / "expE_crossfamily_human.json", "w"), indent=1, ensure_ascii=False)
