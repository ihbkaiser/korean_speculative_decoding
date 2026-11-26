"""E-B: is the margin result a tautology? Use the margin of a model that is NOT in the draft-target pair."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ksdlib import *
from sklearn.metrics import roc_auc_score
SIZE = {"qwen06": "0.6B", "qwen17": "1.7B", "qwen4": "4B", "qwen8": "8B"}
out = []
def analyse(tag, d, judges):
    x = d[d.ptype != "excluded"]
    rec = {"setting": tag, "n": int(len(d)), "rej_rate": round(float(d.rej.mean()), 4),
           "AUC boundary label": round(roc_auc_score(x.rej, (x.ptype != "internal").astype(int)), 3),
           "AUC target margin": round(roc_auc_score(d.rej, -d.mT), 3), "AUC target entropy": round(roc_auc_score(d.rej, d.HT), 3)}
    def ratios(by):
        r = {}
        for nm, (a, b, col) in {"morph/int": ("morph_onset", "internal", "ptype"), "word/int": ("word_onset", "internal", "ptype"), "fused/split": ("CROSS", "WITHIN_SPLIT", "aclass")}.items():
            m, raw = std_ratio(d, a, b, by, col=col); r[nm] = [round(raw, 2), round(m, 2)]
        return r
    rec["matched on target margin"] = ratios("mT")
    rec["matched on target entropy"] = ratios("HT")
    for j in judges:
        rec[f"AUC judge {SIZE[j]} margin"] = round(roc_auc_score(d.rej, -d["m_" + j]), 3)
        rec[f"matched on judge {SIZE[j]} margin"] = ratios("m_" + j)
        dp = d["m_" + j] < 1
        rec[f"judge {SIZE[j]}: DP share / share of rejections"] = [round(float(dp.mean()), 3), round(float(d.rej[dp].sum() / d.rej.sum()), 3)]
    out.append(rec); print(json.dumps(rec, ensure_ascii=False), flush=True)

# generated text: verified k=4 proposals
for pid, judges in [("Q5", ["qwen4", "qwen17"]), ("Q6", ["qwen4", "qwen06"]), ("Q3", ["qwen17", "qwen06"]), ("Q2", ["qwen17"]), ("Q4", ["qwen06"])]:
    tgt, drf, _ = PAIRS[pid]
    df = load_pair("generated", tgt, drf)
    for j in judges:
        jd = read_tf("generated", tgt, j, cols=["pool_index", "position", "top2_margin"])
        assert len(jd) == len(df); df["m_" + j] = jd.top2_margin.values
    split = "all" if tgt == "qwen4" else "common"
    al = read_aligned(tgt, split)
    df = df.merge(al, on=["doc", "pos"], how="left"); df["ptype"] = df.ptype.fillna("excluded"); df["aclass"] = df.aclass.fillna("OTHER")
    ev = pd.read_parquet(f"{ROOT}/simulated/{pid}/common/events_k4.parquet", columns=["doc_id", "position", "eligible", "rejected"])
    ev = ev[ev.eligible]
    ids = read_tf("generated", tgt, tgt, cols=["doc_id", "pool_index", "position"]).drop_duplicates("doc_id")
    ev["doc"] = ev.doc_id.map(dict(zip(ids.doc_id, ids.pool_index))); ev = ev.dropna(subset=["doc"]); ev["doc"] = ev.doc.astype(np.int64); ev["pos"] = ev.position.astype(np.int32)
    d = df.merge(ev[["doc", "pos", "rejected"]], on=["doc", "pos"], how="inner"); d["rej"] = d.rejected.astype(int)
    analyse(f"generated {pid} ({SIZE[drf]}->{SIZE[tgt]})", d, judges)

# human-written text: every position, disagreement between draft and target argmax
for drf, tgt, judges in [("qwen06", "qwen4", ["qwen8", "qwen17"]), ("qwen17", "qwen4", ["qwen8", "qwen06"]), ("qwen06", "qwen8", ["qwen4", "qwen17"]), ("qwen4", "qwen8", ["qwen17", "qwen06"])]:
    host = "qwen8"  # all four Qwen3 models scored the same human continuation under this directory
    t = read_tf("human", host, tgt); dd = read_tf("human", host, drf)
    assert (t.position.values == dd.position.values).all()
    d = pd.DataFrame({"doc": t.pool_index.values.astype(np.int64), "pos": t.position.values.astype(np.int32), "mT": t.top2_margin.values, "HT": t.entropy.values,
                      "rej": (t.argmax_id.values != dd.argmax_id.values).astype(int)})
    for j in judges:
        d["m_" + j] = read_tf("human", host, j, cols=["pool_index", "position", "top2_margin"]).top2_margin.values
    al = read_aligned(host, "human")
    d = d.merge(al, on=["doc", "pos"], how="left"); d["ptype"] = d.ptype.fillna("excluded"); d["aclass"] = d.aclass.fillna("OTHER")
    analyse(f"human {SIZE[drf]}->{SIZE[tgt]}", d, judges)
json.dump(out, open(OUT / "expB_thirdparty.json", "w"), indent=1, ensure_ascii=False)
