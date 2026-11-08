"""End-to-end analysis: load -> features -> quality gate -> LOSO models -> tables and figures.

Usage::

    python scripts/download_data.py          # once
    python scripts/run_all.py [--data data/sleep-accel] [--out results] [--quick]

Only aggregated tables, per-participant summary rows and figures are written to
``results/``. Per-epoch caches go to ``data/cache/`` (git-ignored).
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sleepstage import io  # noqa: E402
from sleepstage.evaluate import (  # noqa: E402
    bland_altman,
    evaluate_predictions,
    loso_predict,
    night_level_table,
)
from sleepstage.features import HR_BASE, MOTION_BASE, build_features, feature_columns  # noqa: E402
from sleepstage.models import MODELS, TASK_LABELS, TASKS, make_model, task_labels  # noqa: E402
from sleepstage.quality import FLAGS, nominal_rate, quality_flags, summarise_flags  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
# palette (reference instance of the dataviz skill): categorical slots 1-3 + neutral ink
C_PSG, C_PRED, C_THIRD = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "figure.dpi": 130,
        "savefig.bbox": "tight",
    }
)
SCENARIOS = {
    # name: (train on, evaluate on)
    "all_epochs": ("all", "all"),
    "reliable_eval_only": ("all", "reliable"),
    "gated": ("reliable", "reliable"),
}


# ----------------------------------------------------------------------------- data
def load_epochs(data: Path, cache: Path, subjects: list[str] | None) -> pd.DataFrame:
    key = cache / f"epochs_{'all' if subjects is None else len(subjects)}.pkl"
    if key.exists():
        return pd.read_pickle(key)
    frames = []
    ids = io.list_subjects(data) if subjects is None else subjects
    for i, sid in enumerate(ids, 1):
        night = io.load_night(data, sid)
        df = build_features(night).join(quality_flags(night))
        df["acc_rate_hz"] = nominal_rate(night.acc_t)
        frames.append(df.reset_index())
        print(f"  loaded {sid} ({i}/{len(ids)})", flush=True)
    out = pd.concat(frames, ignore_index=True)
    cache.mkdir(parents=True, exist_ok=True)
    out.to_pickle(key)
    return out


def participant_summary(ep: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid, g in ep.groupby("sid", sort=False):
        sc = g["stage"] >= 0
        r = {"sid": sid, "n_epochs_scored": int(sc.sum()), "n_unscored": int((~sc).sum())}
        for code, name in [(0, "W"), (1, "N1"), (2, "N2"), (3, "N3"), (5, "REM")]:
            r[f"min_{name}"] = float((g["stage"] == code).sum() * 0.5)
        r["acc_rate_hz"] = round(float(g["acc_rate_hz"].iloc[0]), 1)
        r["hr_samples_per_epoch_median"] = float(g.loc[sc, "hr_n"].median())
        r.update({k: round(v, 2) for k, v in summarise_flags(g, sc).items() if k != "n_epochs"})
        rows.append(r)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- modelling
def run_models(ep, feats, models, tasks, n_jobs, label=""):
    """LOSO predictions for each (task, model, train set). Returns dict keyed by tuple."""
    X = ep[feats].to_numpy(float)
    groups = ep["sid"].to_numpy()
    rel = ep["reliable"].to_numpy(bool)
    preds = {}
    for task in tasks:
        y = task_labels(ep["stage"].to_numpy(), task)
        for model in models:
            for train in ("all", "reliable"):
                t = time.time()
                preds[(task, model, train)] = loso_predict(
                    X, y, groups, model,
                    train_mask=rel if train == "reliable" else None,
                    model_factory=lambda n: make_model(n, n_jobs=n_jobs),
                )
                print(f"  {label}{task:20s} {model:6s} train={train:8s} {time.time() - t:5.1f}s", flush=True)
    return preds


def scenario_results(ep, preds, tasks, models):
    rel = ep["reliable"].to_numpy(bool)
    groups = ep["sid"].to_numpy()
    rows, per_subj, cms = [], [], {}
    for task in tasks:
        y = task_labels(ep["stage"].to_numpy(), task)
        scored = y >= 0
        for model in models:
            for scen, (train, evalset) in SCENARIOS.items():
                mask = rel if evalset == "reliable" else np.ones(len(y), bool)
                res = evaluate_predictions(y, preds[(task, model, train)], groups, task, mask)
                cms[(task, model, scen)] = res.pop("confusion")
                k = res.pop("per_subject")
                per_subj.append(
                    pd.DataFrame({"task": task, "model": model, "scenario": scen, "sid": k.index, "kappa": k.values})
                )
                excluded = 100.0 * (1 - (mask & scored).sum() / scored.sum())
                rows.append({"task": task, "model": model, "scenario": scen, "pct_epochs_excluded": excluded, **res})
    return pd.DataFrame(rows), pd.concat(per_subj, ignore_index=True), cms


# ----------------------------------------------------------------------------- figures
def plot_confusions(cms, tasks, models, out: Path):
    fig, axes = plt.subplots(len(models), len(tasks), figsize=(3.4 * len(tasks), 3.1 * len(models)))
    axes = np.atleast_2d(axes)
    for j, task in enumerate(tasks):
        names = TASKS[task][0]
        for i, model in enumerate(models):
            cm = cms[(task, model, "all_epochs")]
            frac = cm / cm.sum(axis=1, keepdims=True)
            ax = axes[i, j]
            ax.grid(False)
            ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
            for a in range(len(names)):
                for b in range(len(names)):
                    ax.text(b, a, f"{frac[a, b]:.2f}\n({cm[a, b]})", ha="center", va="center", fontsize=7,
                            color="white" if frac[a, b] > 0.6 else INK)
            ax.set_xticks(range(len(names)), names, rotation=30)
            ax.set_yticks(range(len(names)), names)
            ax.set_xlabel("predicted")
            ax.set_ylabel("PSG")
            ax.set_title(f"{TASK_LABELS[task]}\n{model}, all epochs", fontsize=9, color=INK)
    fig.suptitle("LOSO confusion matrices (row-normalised; counts in brackets)", color=INK)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_bland_altman(night: pd.DataFrame, model: str, out: Path):
    metrics = [("TST", "TST (min)"), ("SE", "SE (%)"), ("WASO", "WASO (min)")]
    strategies = [("no_gate", "no gate"), ("impute", "gate + impute"), ("exclude", "gate + exclude")]
    fig, axes = plt.subplots(3, 3, figsize=(10, 8.5), sharey="row")
    for i, (m, lab) in enumerate(metrics):
        for j, (s, slab) in enumerate(strategies):
            ax = axes[i, j]
            ref, dev = night[f"{m}_psg"], night[f"{m}_{s}"]
            ba = bland_altman(dev, ref)
            mean = (ref + dev) / 2
            heavy = night["pct_unreliable"] > 10
            ax.scatter(mean[~heavy], (dev - ref)[~heavy], s=22, color=C_PSG, edgecolor="white", linewidth=0.8,
                       label="participant", zorder=3)
            ax.scatter(mean[heavy], (dev - ref)[heavy], s=30, marker="D", color=C_PRED, edgecolor="white",
                       linewidth=0.8, label=">10 % unreliable epochs", zorder=3)
            ax.axhline(ba["bias"], color=INK, lw=1.2)
            for v in (ba["loa_low"], ba["loa_high"]):
                ax.axhline(v, color=INK2, lw=1, ls="--")
            ax.axhline(0, color=GRID, lw=1, zorder=1)
            ax.text(0.02, 0.97, f"bias {ba['bias']:+.1f}\nLoA [{ba['loa_low']:.1f}, {ba['loa_high']:.1f}]",
                    transform=ax.transAxes, va="top", fontsize=7.5, color=INK, zorder=5,
                    bbox=dict(facecolor="white", edgecolor=GRID, boxstyle="round,pad=0.3"))
            ax.set_xlabel(f"mean of PSG and device, {lab}")
            if j == 0:
                ax.set_ylabel(f"device - PSG, {lab}")
            if i == 0:
                ax.set_title(slab, color=INK)
    axes[0, 0].legend(loc="lower left", fontsize=7, frameon=False)
    fig.suptitle(f"Bland-Altman agreement with PSG ({model}, sleep vs wake, LOSO)", color=INK)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_hypnogram(ep, pred, sid, kappa, out: Path):
    g = ep[ep["sid"] == sid]
    p = pred[(ep["sid"] == sid).to_numpy()]
    y = task_labels(g["stage"].to_numpy(), "wake_light_deep_rem")
    level = {0: 3, 3: 2, 1: 1, 2: 0}  # wake top, REM, light, deep
    names = ["deep", "light", "REM", "wake"]
    h = g["hours"].to_numpy()
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.2), sharex=True)
    for ax, lab, color, title in [(axes[0], y, C_PSG, "PSG"), (axes[1], p, C_PRED, "predicted (rf, all epochs)")]:
        v = np.array([level.get(int(c), np.nan) if c >= 0 else np.nan for c in lab], float)
        ax.step(h, v, where="post", color=color, lw=1.6)
        unrel = ~g["reliable"].to_numpy(bool)
        if unrel.any():
            ax.fill_between(h, -0.5, 3.5, where=unrel, step="post", color=GRID, alpha=0.9, lw=0,
                            label="unreliable epoch (quality gate)")
        ax.set_yticks(range(4), names)
        ax.set_ylim(-0.5, 3.5)
        ax.set_title(title, loc="left", fontsize=9, color=INK)
    axes[1].set_xlabel("hours since recording start")
    if (~g["reliable"]).any():
        axes[0].legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), fontsize=7, frameon=False)
    fig.suptitle(f"Example hypnogram, participant with median LOSO kappa (4-class rf, kappa = {kappa:.2f})",
                 color=INK)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_gate_effect(table: pd.DataFrame, out: Path):
    tasks = list(TASKS)
    models = list(table["model"].unique())
    fig, ax = plt.subplots(figsize=(8, 3.4))
    width = 0.13
    colors = {"all_epochs": C_PSG, "reliable_eval_only": C_THIRD, "gated": C_PRED}
    labels = {"all_epochs": "all epochs", "reliable_eval_only": "trained all, evaluated reliable",
              "gated": "trained + evaluated reliable"}
    x = 0
    ticks, ticklabels = [], []
    for task in tasks:
        for model in models:
            for k, scen in enumerate(SCENARIOS):
                r = table[(table.task == task) & (table.model == model) & (table.scenario == scen)].iloc[0]
                ax.bar(x + k * width, r["kappa_pooled"], width - 0.02, color=colors[scen],
                       label=labels[scen] if x == 0 else None)
            ticks.append(x + width)
            ticklabels.append(f"{TASK_LABELS[task].split(' ')[0]} {model}")
            x += 0.6
    ax.set_xticks(ticks, ticklabels)
    ax.set_ylabel("pooled Cohen's kappa")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper right")
    ax.set_ylim(0, max(0.8, table["kappa_pooled"].max() + 0.1))
    ax.set_title("Effect of the quality gate on epoch-level kappa (LOSO)", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def fmt_md(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = [format(v, floatfmt) if isinstance(v, (float, np.floating)) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "sleep-accel")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    ap.add_argument("--cache", type=Path, default=ROOT / "data" / "cache")
    ap.add_argument("--quick", action="store_true", help="6 participants, for smoke-testing")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--reuse-predictions", action="store_true",
                    help="reuse cached LOSO predictions from --cache (skips refitting)")
    args = ap.parse_args(argv)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    subjects = io.list_subjects(args.data)[:6] if args.quick else None
    print("Loading data and computing features ...")
    ep = load_epochs(args.data, args.cache, subjects)
    feats = feature_columns()
    tasks, models = list(TASKS), MODELS

    # --- dataset / quality summaries
    ps = participant_summary(ep)
    ps.to_csv(out / "participants.csv", index=False)
    scored = ep["stage"] >= 0
    qual = summarise_flags(ep, scored)
    qual["n_participants"] = int(ep["sid"].nunique())
    qual["n_participants_gt10pct_unreliable"] = int((ps["pct_unreliable"] > 10).sum())
    pd.DataFrame([qual]).to_csv(out / "quality_summary.csv", index=False)
    print(json.dumps(qual, indent=1))
    # PSG stage composition of reliable vs unreliable scored epochs
    sc = ep[scored]
    comp = pd.crosstab(sc["reliable"].map({True: "reliable", False: "unreliable"}),
                       sc["stage"].map(io.STAGE_NAMES), normalize="index") * 100
    comp["n_epochs"] = sc["reliable"].map({True: "reliable", False: "unreliable"}).value_counts()
    comp.round(1).to_csv(out / "quality_stage_composition.csv")

    # --- main models
    print("LOSO models ...")
    pred_file = args.cache / f"predictions_{'quick' if args.quick else 'all'}.pkl"
    preds = {}
    if args.reuse_predictions and pred_file.exists():
        with open(pred_file, "rb") as f:
            preds = pickle.load(f)
        print(f"  reusing {len(preds)} cached prediction sets")
    if not all((t, m, tr) in preds for t in tasks for m in models for tr in ("all", "reliable")):
        preds.update(run_models(ep, feats, models, tasks, args.n_jobs))

    def save_preds():  # per-epoch predictions stay in the git-ignored cache
        args.cache.mkdir(parents=True, exist_ok=True)
        with open(pred_file, "wb") as f:
            pickle.dump(preds, f)

    save_preds()
    table, per_subj, cms = scenario_results(ep, preds, tasks, models)
    per_subj.to_csv(out / "per_participant_kappa.csv", index=False)
    cols = ["task", "model", "scenario", "pct_epochs_excluded", "n_epochs", "kappa_pooled",
            "kappa_subject_mean", "kappa_subject_sd", "kappa_subject_median", "accuracy"]
    sens_cols = [c for c in table.columns if c.startswith("sens_")] + ["wake_specificity"]
    table[cols + sens_cols].to_csv(out / "epoch_metrics.csv", index=False)

    # kappa table: one row per task x model; columns per scenario + delta
    kt = table.pivot_table(index=["task", "model"], columns="scenario", values="kappa_pooled").reset_index()
    excl = table[table.scenario == "gated"].set_index(["task", "model"])["pct_epochs_excluded"]
    kt["pct_excluded"] = [excl[(t, m)] for t, m in zip(kt.task, kt.model)]
    kt["delta_kappa_gated"] = kt["gated"] - kt["all_epochs"]
    kt["delta_kappa_eval_only"] = kt["reliable_eval_only"] - kt["all_epochs"]
    kt = kt[["task", "model", "all_epochs", "reliable_eval_only", "gated", "delta_kappa_gated",
             "delta_kappa_eval_only", "pct_excluded"]]
    kt.to_csv(out / "kappa_table.csv", index=False)
    (out / "kappa_table.md").write_text(fmt_md(kt))

    allep = table[table.scenario == "all_epochs"][["task", "model", "kappa_pooled", "kappa_subject_mean",
                                                   "kappa_subject_sd", "accuracy"] + sens_cols]
    (out / "epoch_metrics_all_epochs.md").write_text(fmt_md(allep))

    for (task, model, scen), cm in cms.items():
        if scen == "all_epochs":
            names = TASKS[task][0]
            pd.DataFrame(cm, index=[f"psg_{n}" for n in names], columns=[f"pred_{n}" for n in names]).to_csv(
                out / f"confusion_{task}_{model}.csv")
    plot_confusions(cms, tasks, models, out / "confusion_matrices.png")
    plot_gate_effect(table, out / "quality_gate_kappa.png")

    # --- night-level agreement (sleep vs wake)
    agree_rows = []
    for model in models:
        night = night_level_table(ep, preds[("sleep_wake", model, "all")], preds[("sleep_wake", model, "reliable")])
        night.round(2).to_csv(out / f"night_level_{model}.csv", index=False)
        subsets = {"all nights": night, "nights <=50% unreliable": night[night["pct_unreliable"] <= 50]}
        for subset, nd in subsets.items():
            for m in ["TST", "SE", "WASO"]:
                for s in ["no_gate", "impute", "exclude"]:
                    agree_rows.append({"model": model, "nights": subset, "metric": m, "strategy": s,
                                       **bland_altman(nd[f"{m}_{s}"], nd[f"{m}_psg"])})
        plot_bland_altman(night, model, out / f"bland_altman_{model}.png")
    agree = pd.DataFrame(agree_rows)
    agree.to_csv(out / "night_level_agreement.csv", index=False)
    (out / "night_level_agreement.md").write_text(fmt_md(agree.drop(columns=["n"]), ".1f"))

    # --- example hypnogram: participant whose 4-class rf kappa is closest to the median
    k4 = per_subj[(per_subj.task == "wake_light_deep_rem") & (per_subj.model == "rf")
                  & (per_subj.scenario == "all_epochs")].set_index("sid")["kappa"]
    sid_med = (k4 - k4.median()).abs().idxmin()
    plot_hypnogram(ep, preds[("wake_light_deep_rem", "rf", "all")], sid_med, k4[sid_med], out / "hypnogram_example.png")

    # --- feature-set ablation (rf, trained and evaluated on all epochs)
    print("Feature ablation (rf) ...")
    sets = {
        "motion": [c for c in feats if c.split("_")[0] in MOTION_BASE],
        "hr": [c for c in feats if c.split("_")[0] in HR_BASE],
        "motion+hr (no time)": feature_columns(include_time=False),
        "motion+hr+time": feats,
    }
    abl = []
    groups = ep["sid"].to_numpy()
    for name, cols_ in sets.items():
        for task in tasks:
            y = task_labels(ep["stage"].to_numpy(), task)
            if name == "motion+hr+time":
                p = preds[(task, "rf", "all")]
            elif ("ablation", name, task) in preds:
                p = preds[("ablation", name, task)]
            else:
                p = loso_predict(ep[cols_].to_numpy(float), y, groups, "rf",
                                 model_factory=lambda n: make_model(n, n_jobs=args.n_jobs))
                preds[("ablation", name, task)] = p
                save_preds()
            r = evaluate_predictions(y, p, groups, task, np.ones(len(y), bool))
            abl.append({"features": name, "task": task, "kappa_pooled": r["kappa_pooled"],
                        "kappa_subject_mean": r["kappa_subject_mean"], "accuracy": r["accuracy"]})
            print(f"  {name:22s} {task:20s} kappa={r['kappa_pooled']:.3f}", flush=True)
    abl = pd.DataFrame(abl)
    abl.to_csv(out / "feature_ablation_rf.csv", index=False)
    (out / "feature_ablation_rf.md").write_text(fmt_md(abl))

    summary = {
        "n_participants": int(ep["sid"].nunique()),
        "n_scored_epochs": int(scored.sum()),
        "quality": qual,
        "hypnogram_participant": str(sid_med),
        "runtime_min": round((time.time() - t_start) / 60, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    save_preds()
    print(f"Done in {summary['runtime_min']} min. Results in {out}")


if __name__ == "__main__":
    main()
