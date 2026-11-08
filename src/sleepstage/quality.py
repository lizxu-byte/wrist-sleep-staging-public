"""Epoch-level data-quality gate for wrist HR + accelerometer data.

The Apple Watch in this dataset delivers *processed* heart rate (bpm, roughly
every 5 s), not the raw PPG waveform, so classical PPG signal-quality indices
(perfusion, template correlation, motion-artefact detection on the waveform)
cannot be computed. The gate therefore checks what *is* observable:

``hr_sparse``
    fewer than ``min_hr_samples`` HR samples inside the 30 s epoch (nominal ~6).
    Covers missing HR and long sampling gaps.
``hr_implausible``
    any in-epoch HR sample outside ``[hr_min, hr_max]`` bpm.
``hr_jump``
    any change > ``max_jump_bpm`` between consecutive HR samples that are at
    most ``jump_window_s`` apart, where the later sample lies in the epoch.
    Physiological HR accelerations during arousals rarely exceed this; values
    much larger point to an optical tracking error.
``acc_gap``
    fewer than ``min_acc_fraction`` of the expected accelerometer samples in the
    epoch. The expected count uses the night's median sampling interval.

An epoch is ``reliable`` if no flag is set. Unreliable epochs are not forced
into a stage: the downstream analysis either excludes them or imputes them
explicitly (see :mod:`sleepstage.evaluate`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .io import EPOCH_S, Night, assign_epochs

FLAGS = ["hr_sparse", "hr_implausible", "hr_jump", "acc_gap"]


@dataclass(frozen=True)
class QualityConfig:
    min_hr_samples: int = 2
    hr_min: float = 30.0
    hr_max: float = 200.0
    max_jump_bpm: float = 30.0
    jump_window_s: float = 10.0
    min_acc_fraction: float = 0.5


def nominal_rate(t: np.ndarray) -> float:
    """Median sampling rate in Hz (NaN if fewer than two samples)."""
    if len(t) < 2:
        return float("nan")
    dt = np.diff(t)
    dt = dt[dt > 0]
    return float(1.0 / np.median(dt)) if len(dt) else float("nan")


def quality_flags(night: Night, cfg: QualityConfig = QualityConfig()) -> pd.DataFrame:
    n = night.n_epochs
    t0 = night.epoch_start[0]
    out = pd.DataFrame(index=pd.RangeIndex(n, name="epoch"))

    # --- heart rate ---
    e_hr = assign_epochs(night.hr_t, t0, n)
    ok = e_hr >= 0
    hr_n = np.bincount(e_hr[ok], minlength=n) if ok.any() else np.zeros(n, int)
    out["hr_sparse"] = hr_n < cfg.min_hr_samples

    bad = ok & ((night.hr < cfg.hr_min) | (night.hr > cfg.hr_max))
    out["hr_implausible"] = np.bincount(e_hr[bad], minlength=n) > 0 if bad.any() else False

    jump = np.zeros(n, bool)
    if len(night.hr) > 1:
        dhr = np.abs(np.diff(night.hr))
        dt = np.diff(night.hr_t)
        j = (dhr > cfg.max_jump_bpm) & (dt <= cfg.jump_window_s)
        e_later = e_hr[1:][j]
        e_later = e_later[e_later >= 0]
        jump[e_later] = True
    out["hr_jump"] = jump

    # --- accelerometer ---
    e_acc = assign_epochs(night.acc_t, t0, n)
    acc_n = np.bincount(e_acc[e_acc >= 0], minlength=n) if (e_acc >= 0).any() else np.zeros(n)
    rate = nominal_rate(night.acc_t)
    expected = rate * EPOCH_S if np.isfinite(rate) else np.inf
    out["acc_gap"] = acc_n < cfg.min_acc_fraction * expected

    out["reliable"] = ~out[FLAGS].any(axis=1)
    return out


def summarise_flags(flags: pd.DataFrame, scored: pd.Series | None = None) -> dict[str, float]:
    """Share (%) of epochs carrying each flag, optionally restricted to scored epochs."""
    f = flags if scored is None else flags[scored.to_numpy()]
    res = {f"pct_{c}": 100.0 * float(f[c].mean()) for c in FLAGS}
    res["pct_unreliable"] = 100.0 * float((~f["reliable"]).mean())
    res["n_epochs"] = int(len(f))
    return res
