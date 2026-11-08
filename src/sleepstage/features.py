"""Per-epoch features computed only from signals a wrist wearable has.

Motion (from raw tri-axial acceleration, in g)
---------------------------------------------
For every sample the vector magnitude ``r = sqrt(x^2 + y^2 + z^2)`` is computed.
Per 30 s epoch:

* ``enmo``  - mean of ``max(r - 1 g, 0)`` in milli-g (Euclidean Norm Minus One,
  van Hees et al. 2013). No band-pass filter is applied.
* ``mad``   - mean amplitude deviation, mean ``|r - mean(r)|`` within the epoch, in
  milli-g. Insensitive to calibration offsets.
* ``angle`` - arm-angle change (van Hees et al. 2015 style): the z-angle
  ``atan(z / sqrt(x^2 + y^2))`` in degrees is averaged over 5 s blocks aligned to
  the epoch grid; the epoch value is the mean absolute change between
  consecutive 5 s blocks whose later block lies in the epoch.

All three are means over samples, so they do not depend on the sampling rate,
which varies between participants (10-66 Hz). We deliberately do not reproduce
the te Lindert & Van Someren activity-count conversion used by Walch et al.
Epochs without accelerometer samples are linearly interpolated across epochs
(nearest value at the edges) and flagged by :mod:`sleepstage.quality`.

Heart rate (Apple Watch processed HR, bpm)
------------------------------------------
* ``hr``    - epoch HR (mean of in-epoch samples, time-interpolated if none; see
  :func:`sleepstage.io.epoch_heart_rate`).
* ``hrsd``  - SD of the HR samples within the epoch (needs >= 2 samples; otherwise
  interpolated across epochs). This is a crude short-term variability measure;
  beat-to-beat HRV is not available from this device output.

Normalisation
-------------
Motion features and ``hrsd`` are ``log1p``-transformed. Then every base feature
is robust-z-scored per night: ``(x - median) / IQR``, with median and IQR taken
over that participant's epochs with *observed* data (interpolated epochs are not
used for the statistics). This uses the whole night, i.e. it is an offline,
"after waking up" computation, which matches how consumer devices report sleep.

Context windows
---------------
For each normalised base feature ``f``: ``f_mean5``/``f_sd5`` (rolling mean/SD
over +-5 min, i.e. 21 epochs) and ``f_mean15`` (rolling mean over +-15 min, 61
epochs). Windows are **centred** by default (they use future epochs, again an
offline choice); ``mode="causal"`` uses the same number of epochs ending at the
current epoch. Windows are truncated at the start and end of the night
(``min_periods=1`` for means, 2 for SDs).

Time feature
------------
``hours`` = hours since the first epoch of the recording. In this dataset the
recording (PSG scoring) starts close to lights-off in the lab, so this feature
is more informative than a wearable's "time since tracking started" would be.
We do **not** use PSG lights-off/lights-on markers (they are not in the
dataset); an ablation without ``hours`` is reported by ``scripts/run_all.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .io import EPOCH_S, Night, assign_epochs, epoch_table

BLOCK_S = 5.0
BASE_FEATURES = ["enmo", "mad", "angle", "hr", "hrsd"]
MOTION_BASE = ["enmo", "mad", "angle"]
HR_BASE = ["hr", "hrsd"]
SHORT_HALF_WIDTH = 10  # epochs: +-5 min
LONG_HALF_WIDTH = 30  # epochs: +-15 min


def motion_epoch_metrics(night: Night) -> pd.DataFrame:
    """Per-epoch ENMO, MAD, arm-angle change (NaN where no samples) and sample count."""
    n = night.n_epochs
    t0 = night.epoch_start[0]
    out = pd.DataFrame(index=pd.RangeIndex(n, name="epoch"))
    idx = assign_epochs(night.acc_t, t0, n)
    ok = idx >= 0
    acc = night.acc[ok]
    e = idx[ok]
    out["acc_n"] = np.bincount(e, minlength=n).astype(int) if len(e) else np.zeros(n, int)
    if len(e) == 0:
        for c in MOTION_BASE:
            out[c] = np.nan
        return out

    r = np.sqrt((acc**2).sum(axis=1))
    s = pd.DataFrame({"e": e, "r": r, "enmo": np.maximum(r - 1.0, 0.0) * 1000.0})
    g = s.groupby("e")
    out["enmo"] = g["enmo"].mean().reindex(out.index)
    s["dev"] = (s["r"] - g["r"].transform("mean")).abs() * 1000.0
    out["mad"] = s.groupby("e")["dev"].mean().reindex(out.index)

    # arm angle: 5 s block means, absolute change between consecutive blocks
    angle = np.degrees(np.arctan2(acc[:, 2], np.sqrt(acc[:, 0] ** 2 + acc[:, 1] ** 2)))
    block = np.floor((night.acc_t[ok] - t0) / BLOCK_S).astype(np.int64)
    bm = pd.Series(angle).groupby(block).mean()
    per_epoch_blocks = int(round(EPOCH_S / BLOCK_S))
    b = bm.index.to_numpy()
    d = np.abs(np.diff(bm.to_numpy()))
    consecutive = np.diff(b) == 1
    later = b[1:][consecutive]
    out["angle"] = (
        pd.Series(d[consecutive]).groupby(later // per_epoch_blocks).mean().reindex(out.index)
    )
    return out


def rolling(x: pd.Series, half_width: int, stat: str = "mean", mode: str = "centred") -> pd.Series:
    """Rolling statistic over ``2*half_width + 1`` epochs.

    ``mode="centred"``: window ``[i - half_width, i + half_width]``.
    ``mode="causal"``:  window ``[i - 2*half_width, i]`` (no future epochs).
    """
    if mode not in {"centred", "causal"}:
        raise ValueError("mode must be 'centred' or 'causal'")
    w = 2 * half_width + 1
    r = x.rolling(w, center=(mode == "centred"), min_periods=1 if stat == "mean" else 2)
    if stat == "mean":
        return r.mean()
    if stat == "sd":
        return r.std(ddof=0)
    raise ValueError("stat must be 'mean' or 'sd'")


def _interp(x: pd.Series) -> pd.Series:
    return x.interpolate(method="linear", limit_direction="both")


def robust_z(x: pd.Series, observed: pd.Series) -> pd.Series:
    ref = x[observed & x.notna()]
    if len(ref) == 0:
        ref = x.dropna()
    med = ref.median()
    iqr = ref.quantile(0.75) - ref.quantile(0.25)
    if not np.isfinite(iqr) or iqr <= 1e-12:
        iqr = ref.std(ddof=0)
    if not np.isfinite(iqr) or iqr <= 1e-12:
        iqr = 1.0
    return (x - med) / iqr


def feature_columns(include_time: bool = True) -> list[str]:
    cols = []
    for f in BASE_FEATURES:
        cols += [f"{f}_z", f"{f}_z_mean5", f"{f}_z_sd5", f"{f}_z_mean15"]
    if include_time:
        cols.append("hours")
    return cols


def build_features(night: Night, mode: str = "centred") -> pd.DataFrame:
    """Epoch table with PSG stage, raw aggregates and model features for one night."""
    df = epoch_table(night).join(motion_epoch_metrics(night))
    raw = pd.DataFrame(index=df.index)
    observed = {
        "enmo": df["acc_n"] > 0,
        "mad": df["acc_n"] > 0,
        "angle": df["acc_n"] > 0,
        "hr": df["hr_n"] > 0,
        "hrsd": df["hr_n"] >= 2,
    }
    raw["enmo"] = np.log1p(df["enmo"])
    raw["mad"] = np.log1p(df["mad"])
    raw["angle"] = np.log1p(df["angle"])
    raw["hr"] = df["hr_interp"]
    raw["hrsd"] = np.log1p(df["hr_sd"])
    for f in BASE_FEATURES:
        z = robust_z(_interp(raw[f]), observed[f])
        if z.isna().all():  # signal entirely absent for this night
            z = pd.Series(0.0, index=df.index)
        df[f"{f}_z"] = z
        df[f"{f}_z_mean5"] = rolling(z, SHORT_HALF_WIDTH, "mean", mode)
        df[f"{f}_z_sd5"] = rolling(z, SHORT_HALF_WIDTH, "sd", mode).fillna(0.0)
        df[f"{f}_z_mean15"] = rolling(z, LONG_HALF_WIDTH, "mean", mode)
    df["hours"] = (df["t_start"] - df["t_start"].iloc[0]) / 3600.0
    return df
