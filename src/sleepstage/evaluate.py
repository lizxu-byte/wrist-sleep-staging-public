"""Leave-one-subject-out (LOSO) evaluation, epoch metrics and night-level agreement.

Conventions
-----------
* All models are trained and tested with LOSO: for each participant, a fresh
  model is fitted on all *other* participants' scored epochs and predicts the
  held-out participant. Per-night feature normalisation (``features.py``) only
  uses the participant's own data, so nothing leaks across folds.
* ``kappa_pooled`` / ``accuracy`` / sensitivities are computed on the
  concatenation of all held-out predictions. ``kappa_subject_mean`` /
  ``_sd`` / ``_median`` summarise the per-participant Cohen's kappa.
* ``wake_specificity`` follows the actigraphy convention (and Walch et al.
  2019): the fraction of PSG-wake epochs scored as wake. In the multi-class
  tasks it equals the wake sensitivity.
* Night-level metrics (sleep vs wake): TIB = number of epochs counted x 0.5 min;
  TST = sleep epochs x 0.5 min; SE = 100 * TST / TIB; sleep onset = first sleep
  epoch, final awakening = last sleep epoch; WASO = wake epochs between them.
  No minimum bout length / persistent-sleep rule is applied.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

from .io import EPOCH_S
from .models import TASKS, make_model, task_labels

EPOCH_MIN = EPOCH_S / 60.0


# ----------------------------------------------------------------------------- LOSO
def loso_splits(groups: Sequence) -> Iterator[tuple[object, np.ndarray, np.ndarray]]:
    """Yield ``(held_out_group, train_idx, test_idx)``; train and test never share a group."""
    groups = np.asarray(groups)
    for g in pd.unique(groups):
        test = groups == g
        train_idx, test_idx = np.flatnonzero(~test), np.flatnonzero(test)
        assert not set(groups[train_idx]) & set(groups[test_idx])
        yield g, train_idx, test_idx


def loso_predict(
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence,
    model: str,
    train_mask: np.ndarray | None = None,
    model_factory=make_model,
) -> np.ndarray:
    """LOSO predictions for every row (labels ``y < 0`` are never used for training).

    ``train_mask`` further restricts the training rows (e.g. reliable epochs only).
    Every row of the held-out participant receives a prediction, so that the
    caller can choose which epochs to evaluate.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    usable = y >= 0
    if train_mask is not None:
        usable &= np.asarray(train_mask, bool)
    pred = np.full(len(y), -1, dtype=int)
    for _, tr, te in loso_splits(groups):
        tr = tr[usable[tr]]
        clf = model_factory(model)
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te])
    return pred


# ----------------------------------------------------------------------------- epoch metrics
def classification_metrics(y: np.ndarray, pred: np.ndarray, class_names: list[str]) -> dict:
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y, pred, labels=labels)
    with np.errstate(invalid="ignore", divide="ignore"):
        sens = np.diag(cm) / cm.sum(axis=1)
    out = {
        "n_epochs": int(len(y)),
        "kappa_pooled": float(cohen_kappa_score(y, pred, labels=labels)),
        "accuracy": float(accuracy_score(y, pred)),
    }
    for name, s in zip(class_names, sens):
        out[f"sens_{name}"] = float(s)
    out["wake_specificity"] = float(sens[0])
    out["confusion"] = cm
    return out


def per_subject_kappa(y, pred, groups, n_classes: int) -> pd.Series:
    df = pd.DataFrame({"y": y, "p": pred, "g": groups})
    labels = list(range(n_classes))
    res = {}
    for g, d in df.groupby("g", sort=False):
        if len(d) == 0:
            continue
        if d["y"].nunique() == 1 and d["p"].nunique() == 1:
            res[g] = 1.0 if d["y"].iloc[0] == d["p"].iloc[0] else 0.0
        else:
            res[g] = cohen_kappa_score(d["y"], d["p"], labels=labels)
    return pd.Series(res, name="kappa", dtype=float)


def evaluate_predictions(y, pred, groups, task: str, mask: np.ndarray) -> dict:
    """Pooled metrics plus per-subject kappa summary on the epochs selected by ``mask``."""
    names = TASKS[task][0]
    mask = np.asarray(mask, bool) & (np.asarray(y) >= 0)
    y, pred, groups = np.asarray(y)[mask], np.asarray(pred)[mask], np.asarray(groups)[mask]
    out = classification_metrics(y, pred, names)
    k = per_subject_kappa(y, pred, groups, len(names))
    out.update(
        kappa_subject_mean=float(k.mean()),
        kappa_subject_sd=float(k.std(ddof=1)),
        kappa_subject_median=float(k.median()),
        per_subject=k,
    )
    return out


# ----------------------------------------------------------------------------- night level
def night_metrics(sleep: np.ndarray, valid: np.ndarray | None = None) -> dict[str, float]:
    """TST/TIB/WASO/SOL in minutes and SE in % from a time-ordered sleep (1) / wake (0) series.

    Epochs with ``valid == False`` are ignored (not counted as wake or sleep).
    """
    sleep = np.asarray(sleep)
    valid = np.ones(len(sleep), bool) if valid is None else np.asarray(valid, bool)
    s = sleep[valid].astype(int)
    tib = len(s) * EPOCH_MIN
    tst = s.sum() * EPOCH_MIN
    if s.sum() == 0:
        return {"TIB": tib, "TST": 0.0, "SE": 0.0, "WASO": 0.0, "SOL": tib}
    idx = np.flatnonzero(s == 1)
    first, last = idx[0], idx[-1]
    waso = (s[first : last + 1] == 0).sum() * EPOCH_MIN
    return {
        "TIB": tib,
        "TST": tst,
        "SE": 100.0 * tst / tib if tib else np.nan,
        "WASO": waso,
        "SOL": first * EPOCH_MIN,
    }


def impute_nearest(labels: np.ndarray, reliable: np.ndarray) -> np.ndarray:
    """Replace labels of unreliable epochs by the label of the nearest reliable epoch in time.

    Ties go to the earlier epoch. If no epoch is reliable the input is returned unchanged.
    """
    labels = np.asarray(labels).copy()
    reliable = np.asarray(reliable, bool)
    ok = np.flatnonzero(reliable)
    if len(ok) == 0 or len(ok) == len(labels):
        return labels
    bad = np.flatnonzero(~reliable)
    pos = np.searchsorted(ok, bad)
    left = ok[np.clip(pos - 1, 0, len(ok) - 1)]
    right = ok[np.clip(pos, 0, len(ok) - 1)]
    use_left = (pos > 0) & ((pos == len(ok)) | (bad - left <= right - bad))
    labels[bad] = np.where(use_left, labels[left], labels[right])
    return labels


def bland_altman(device: Sequence[float], reference: Sequence[float]) -> dict[str, float]:
    """Mean bias (device - reference) and 95 % limits of agreement (bias +- 1.96 SD)."""
    d = np.asarray(device, float) - np.asarray(reference, float)
    bias = float(np.mean(d))
    sd = float(np.std(d, ddof=1)) if len(d) > 1 else float("nan")
    return {
        "n": int(len(d)),
        "bias": bias,
        "sd": sd,
        "loa_low": bias - 1.96 * sd,
        "loa_high": bias + 1.96 * sd,
        "mean_abs_error": float(np.mean(np.abs(d))),
    }


def night_level_table(
    df: pd.DataFrame,
    pred_all: np.ndarray,
    pred_gated: np.ndarray,
) -> pd.DataFrame:
    """Per-participant TST/SE/WASO from PSG and from three device strategies.

    ``df`` must contain ``sid``, ``stage`` and ``reliable`` (time-ordered within sid).
    ``pred_all``: sleep(1)/wake(0) predictions of a model trained on all epochs.
    ``pred_gated``: predictions of a model trained on reliable epochs only.

    Strategies: ``no_gate`` (pred_all on all scored epochs), ``impute``
    (pred_gated on reliable epochs, unreliable epochs take the nearest reliable
    prediction) and ``exclude`` (only reliable epochs are counted). The PSG
    reference always uses all scored epochs.
    """
    rows = []
    d = df.assign(_pa=pred_all, _pg=pred_gated)
    d = d[d["stage"] >= 0]
    for sid, g in d.groupby("sid", sort=False):
        psg = task_labels(g["stage"].to_numpy(), "sleep_wake")
        rel = g["reliable"].to_numpy(bool)
        est = {
            "psg": night_metrics(psg),
            "no_gate": night_metrics(g["_pa"].to_numpy()),
            "impute": night_metrics(impute_nearest(g["_pg"].to_numpy(), rel)),
            "exclude": night_metrics(g["_pg"].to_numpy(), valid=rel),
        }
        row = {"sid": sid, "pct_unreliable": 100.0 * (1 - rel.mean())}
        for src, m in est.items():
            for k, v in m.items():
                row[f"{k}_{src}"] = v
        rows.append(row)
    return pd.DataFrame(rows)
