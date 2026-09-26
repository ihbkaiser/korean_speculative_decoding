"""Primary statistical analysis and figure generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any


BIN_ORDER = ["2", "3", "4", "5+"]


def _slope_summary(frame, outcome: str, group_name: str, predictor: str = "misalignment") -> dict[str, Any]:
    import statsmodels.formula.api as smf

    data = frame[[outcome, predictor, "prompt_id"]].dropna()
    data[outcome] = data[outcome].astype("int64")
    result: dict[str, Any] = {"fragmentation_bin": group_name, "outcome": outcome, "n": len(data), "slope": None, "p_value": None, "status": "not_estimable"}
    if len(data) < 5 or data[outcome].nunique() < 2 or data[predictor].nunique() < 2:
        return result
    try:
        fitted = smf.logit(f"{outcome} ~ {predictor}", data=data).fit(
            disp=False,
            cov_type="cluster",
            cov_kwds={"groups": data["prompt_id"]},
        )
        result.update({"slope": float(fitted.params[predictor]), "p_value": float(fitted.pvalues[predictor]), "status": "fit"})
    except Exception as exc:
        result["status"] = f"fit_failed:{type(exc).__name__}"
    return result


def analyze_run(run_dir: str | Path) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import statsmodels.formula.api as smf

    root = Path(run_dir)
    figures_dir = Path("figures") / root.name
    figures_dir.mkdir(parents=True, exist_ok=True)
    events = pd.read_parquet(root / "sd_events.parquet")
    teacher = pd.read_parquet(root / "teacher_forced_tokens.parquet")
    eojeols = pd.read_parquet(root / "eojeols.parquet")

    morphology = eojeols[["prompt_id", "eojeol_index", "fragmentation", "fragmentation_bin", "misalignment", "morpheme_count", "llm_token_count"]]
    events = events.merge(morphology, on=["prompt_id", "eojeol_index"], how="left", suffixes=("", "_eojeol"))
    teacher = teacher.merge(morphology, on=["prompt_id", "eojeol_index"], how="left", suffixes=("", "_eojeol"))
    eligible_events = events[events["fragmentation_bin"].isin(BIN_ORDER)].copy()
    eligible_teacher = teacher[teacher["fragmentation_bin"].isin(BIN_ORDER)].copy()

    summary_rows = []
    for bin_name in BIN_ORDER:
        event_group = eligible_events[eligible_events["fragmentation_bin"] == bin_name]
        teacher_group = eligible_teacher[eligible_teacher["fragmentation_bin"] == bin_name]
        rejection = _slope_summary(event_group, "rejected", bin_name)
        disagreement = _slope_summary(teacher_group.rename(columns={"draft_target_disagreement": "disagreement"}), "disagreement", bin_name)
        summary_rows.append({
            "fragmentation_bin": bin_name,
            "n_sd_proposals": len(event_group),
            "n_verified_sd_proposals": int(event_group["rejected"].notna().sum()) if len(event_group) else 0,
            "rejection_rate": float(event_group["rejected"].mean()) if len(event_group) else None,
            "rejection_misalignment_slope": rejection["slope"],
            "rejection_p_value": rejection["p_value"],
            "rejection_fit_status": rejection["status"],
            "n_teacher_forced_tokens": len(teacher_group),
            "teacher_forced_disagreement_rate": float(teacher_group["draft_target_disagreement"].mean()) if len(teacher_group) else None,
            "disagreement_misalignment_slope": disagreement["slope"],
            "disagreement_p_value": disagreement["p_value"],
            "disagreement_fit_status": disagreement["status"],
        })
    pd.DataFrame(summary_rows).to_csv(root / "fragmentation_bin_summary.csv", index=False)

    event_model_data = eligible_events[[
        "prompt_id", "rejected", "misalignment", "fragmentation", "draft_entropy", "target_entropy", "proposal_position",
    ]].dropna()
    if len(event_model_data):
        event_model_data["rejected"] = event_model_data["rejected"].astype("int64")
    event_model_data.to_csv(root / "sd_logistic_input.csv", index=False)
    model_report = ["SD proposal logistic regression"]
    if len(event_model_data) >= 10 and event_model_data["rejected"].nunique() == 2:
        try:
            model = smf.logit(
                "rejected ~ misalignment + fragmentation + draft_entropy + target_entropy + proposal_position",
                data=event_model_data,
            ).fit(disp=False, cov_type="cluster", cov_kwds={"groups": event_model_data["prompt_id"]})
            model_report.append(model.summary().as_text())
            model_report.append("\nCoefficients and p-values:")
            model_report.append(pd.DataFrame({"coefficient": model.params, "p_value": model.pvalues}).to_csv())
        except Exception as exc:
            model_report.append(f"not estimable: {type(exc).__name__}: {exc}")
    else:
        model_report.append(f"not estimable: n={len(event_model_data)}, outcome_classes={event_model_data['rejected'].nunique() if len(event_model_data) else 0}")
    teacher_model_data = eligible_teacher[[
        "prompt_id", "draft_target_disagreement", "misalignment", "fragmentation", "draft_entropy", "target_entropy", "output_token_position",
    ]].dropna()
    if len(teacher_model_data):
        teacher_model_data["draft_target_disagreement"] = teacher_model_data["draft_target_disagreement"].astype("int64")
    teacher_model_data.to_csv(root / "teacher_forced_logistic_input.csv", index=False)
    model_report.extend(["\nTeacher-forced disagreement logistic regression"])
    if len(teacher_model_data) >= 10 and teacher_model_data["draft_target_disagreement"].nunique() == 2:
        try:
            model = smf.logit(
                "draft_target_disagreement ~ misalignment + fragmentation + draft_entropy + target_entropy + output_token_position",
                data=teacher_model_data,
            ).fit(disp=False, cov_type="cluster", cov_kwds={"groups": teacher_model_data["prompt_id"]})
            model_report.append(model.summary().as_text())
            model_report.append(pd.DataFrame({"coefficient": model.params, "p_value": model.pvalues}).to_csv())
        except Exception as exc:
            model_report.append(f"not estimable: {type(exc).__name__}: {exc}")
    else:
        model_report.append(f"not estimable: n={len(teacher_model_data)}, outcome_classes={teacher_model_data['draft_target_disagreement'].nunique() if len(teacher_model_data) else 0}")
    (root / "logistic_regression.txt").write_text("\n".join(model_report), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for bin_name in BIN_ORDER:
        group = eligible_events[eligible_events["fragmentation_bin"] == bin_name]
        plot_group = group.dropna(subset=["misalignment", "rejected"])
        if len(plot_group):
            axes[0].scatter(plot_group["misalignment"], plot_group["rejected"].astype(int), alpha=0.35, s=12, label=f"{bin_name} (n={len(plot_group)})")
        tg = eligible_teacher[eligible_teacher["fragmentation_bin"] == bin_name]
        if len(tg):
            axes[1].scatter(tg["misalignment"], tg["draft_target_disagreement"].astype(int), alpha=0.35, s=12, label=f"{bin_name} (n={len(tg)})")
    axes[0].set(title="SD proposal rejection", xlabel="Boundary misalignment", ylabel="Rejected (0/1)")
    axes[1].set(title="Teacher-forced draft/target disagreement", xlabel="Boundary misalignment", ylabel="Disagreement (0/1)")
    for axis in axes:
        axis.set_ylim(-0.1, 1.1)
        axis.legend(fontsize=8)
    fig.savefig(figures_dir / "misalignment_outcomes.png", dpi=180)
    plt.close(fig)

    counts = eojeols.groupby(["fragmentation_bin", "misalignment"], dropna=False).size().reset_index(name="eojeol_occurrences")
    counts.to_csv(root / "eojeol_counts.csv", index=False)
    return {"events": len(events), "teacher_forced_tokens": len(teacher), "eojeols": len(eojeols), "figures_dir": str(figures_dir)}
