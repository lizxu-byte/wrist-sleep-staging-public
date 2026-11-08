"""Loading and epoch alignment for the PhysioNet ``sleep-accel`` dataset.

File formats (verified against PhysioNet sleep-accel v1.0.0):

* ``labels/<id>_labeled_sleep.txt``  - ``t stage`` (space separated); ``t`` in
  seconds since PSG start, one row per 30 s epoch. Stage codes: 0 wake, 1 N1,
  2 N2, 3 N3, 5 REM, -1 unscored. Code 4 also occurs (356 epochs, three participants) but is
  not documented on the dataset page; it is treated as N3 (Rechtschaffen & Kales
  stage 4 is merged into N3 under AASM rules). See :data:`STAGE_MAP`.
* ``heart_rate/<id>_heartrate.txt``  - ``t,hr`` (comma separated), bpm, irregular
  sampling (median interval 5 s).
* ``motion/<id>_acceleration.txt``   - ``t x y z`` (space separated), in g,
  approximately 50 Hz (10-66 Hz in some participants).

Epoch grid and alignment
------------------------
The epoch grid is taken from the label file: epoch *k* covers the half-open
interval ``[t_k, t_k + 30 s)`` where ``t_k`` is the label timestamp. Sensor
samples are assigned to the epoch whose interval contains their timestamp
(:func:`assign_epochs`); samples outside the scored window are discarded
(this drops the pre-PSG ambulatory week contained in some files). Timestamps
are sorted and exact duplicates removed, because some motion files contain
short backwards jumps.

Heart rate: the epoch value is the mean of the HR samples inside the epoch.
Epochs with no HR sample get a value linearly interpolated in time between the
nearest samples before and after the epoch midpoint (nearest value at the
edges). These epochs are flagged by :mod:`sleepstage.quality`, so the
interpolation is only used in the "all epochs" analysis.

Unscored epochs (stage -1) are kept on the grid (so that rolling windows see a
continuous time axis) but carry ``stage = -1`` and are excluded from training,
evaluation and the PSG night-level reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

EPOCH_S = 30.0

#: Raw label code -> internal stage code. Internal codes: 0 W, 1 N1, 2 N2, 3 N3, 5 REM, -1 unscored.
STAGE_MAP = {-1: -1, 0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 5}
STAGE_NAMES = {0: "W", 1: "N1", 2: "N2", 3: "N3", 5: "REM", -1: "unscored"}


@dataclass
class Night:
    """One participant's PSG night with raw wearable signals cropped to the scored window."""

    sid: str
    epoch_start: np.ndarray  # (n_epochs,) seconds since PSG start
    stage: np.ndarray  # (n_epochs,) internal stage codes
    hr_t: np.ndarray  # (n_hr,)
    hr: np.ndarray  # (n_hr,) bpm
    acc_t: np.ndarray  # (n_acc,)
    acc: np.ndarray  # (n_acc, 3) g

    @property
    def n_epochs(self) -> int:
        return len(self.epoch_start)

    @property
    def t_end(self) -> float:
        return float(self.epoch_start[-1] + EPOCH_S)


def list_subjects(root: str | Path) -> list[str]:
    """Return participant IDs that have all three required files."""
    root = Path(root)
    ids = []
    for f in sorted((root / "labels").glob("*_labeled_sleep.txt")):
        sid = f.name.split("_")[0]
        if (root / "heart_rate" / f"{sid}_heartrate.txt").exists() and (
            root / "motion" / f"{sid}_acceleration.txt"
        ).exists():
            ids.append(sid)
    return ids


def read_labels(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["t", "stage"], engine="c")
    unknown = set(df["stage"].unique()) - set(STAGE_MAP)
    if unknown:
        raise ValueError(f"Unknown stage codes {unknown} in {path}")
    df["stage"] = df["stage"].map(STAGE_MAP).astype(int)
    return df


def read_heart_rate(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=",", header=None, names=["t", "hr"])


def read_motion(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=" ", header=None, names=["t", "x", "y", "z"])


def _clean_times(df: pd.DataFrame, t0: float, t1: float) -> pd.DataFrame:
    """Crop to [t0, t1), sort by time and drop duplicate timestamps."""
    df = df[(df["t"] >= t0) & (df["t"] < t1)]
    df = df.sort_values("t", kind="stable").drop_duplicates("t", keep="first")
    return df.dropna().reset_index(drop=True)


def build_night(sid: str, labels: pd.DataFrame, hr: pd.DataFrame, acc: pd.DataFrame) -> Night:
    """Assemble a :class:`Night` from already-parsed tables (used by the loader and by tests)."""
    labels = labels.sort_values("t").reset_index(drop=True)
    starts = labels["t"].to_numpy(dtype=float)
    steps = np.diff(starts)
    if len(steps) and not np.allclose(steps, EPOCH_S):
        raise ValueError(f"{sid}: label timestamps are not on a regular {EPOCH_S:.0f} s grid")
    t0, t1 = starts[0], starts[-1] + EPOCH_S
    hr = _clean_times(hr, t0, t1)
    acc = _clean_times(acc, t0, t1)
    return Night(
        sid=sid,
        epoch_start=starts,
        stage=labels["stage"].to_numpy(dtype=int),
        hr_t=hr["t"].to_numpy(dtype=float),
        hr=hr["hr"].to_numpy(dtype=float),
        acc_t=acc["t"].to_numpy(dtype=float),
        acc=acc[["x", "y", "z"]].to_numpy(dtype=float),
    )


def load_night(root: str | Path, sid: str) -> Night:
    root = Path(root)
    return build_night(
        sid,
        read_labels(root / "labels" / f"{sid}_labeled_sleep.txt"),
        read_heart_rate(root / "heart_rate" / f"{sid}_heartrate.txt"),
        read_motion(root / "motion" / f"{sid}_acceleration.txt"),
    )


def assign_epochs(t: np.ndarray, t0: float, n_epochs: int, epoch_s: float = EPOCH_S) -> np.ndarray:
    """Epoch index for each timestamp; -1 for samples outside ``[t0, t0 + n_epochs*epoch_s)``.

    Epoch *k* is the half-open interval ``[t0 + k*epoch_s, t0 + (k+1)*epoch_s)``.
    """
    idx = np.floor((np.asarray(t, dtype=float) - t0) / epoch_s).astype(np.int64)
    idx[(idx < 0) | (idx >= n_epochs)] = -1
    return idx


def epoch_heart_rate(night: Night) -> pd.DataFrame:
    """Per-epoch HR aggregates: mean, SD and sample count of in-epoch samples,
    plus ``hr_interp`` (mean if observed, otherwise time-interpolated at the epoch midpoint)."""
    n = night.n_epochs
    idx = assign_epochs(night.hr_t, night.epoch_start[0], n)
    ok = idx >= 0
    s = pd.DataFrame({"e": idx[ok], "hr": night.hr[ok]})
    g = s.groupby("e")["hr"]
    out = pd.DataFrame(index=pd.RangeIndex(n, name="epoch"))
    out["hr_n"] = g.size().reindex(out.index, fill_value=0).astype(int)
    out["hr_mean"] = g.mean().reindex(out.index)
    out["hr_sd"] = g.std(ddof=0).reindex(out.index)
    out.loc[out["hr_n"] < 2, "hr_sd"] = np.nan
    mid = night.epoch_start + EPOCH_S / 2
    if len(night.hr_t):
        interp = np.interp(mid, night.hr_t, night.hr)
    else:
        interp = np.full(n, np.nan)
    out["hr_interp"] = out["hr_mean"].fillna(pd.Series(interp, index=out.index))
    return out


def epoch_table(night: Night) -> pd.DataFrame:
    """Epoch grid with PSG stage and per-epoch HR aggregates (motion metrics live in features)."""
    df = pd.DataFrame(
        {"sid": night.sid, "t_start": night.epoch_start, "stage": night.stage},
        index=pd.RangeIndex(night.n_epochs, name="epoch"),
    )
    return df.join(epoch_heart_rate(night))
