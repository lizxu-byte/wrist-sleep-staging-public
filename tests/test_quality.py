import numpy as np
import pandas as pd

from sleepstage.io import build_night
from sleepstage.quality import FLAGS, QualityConfig, quality_flags, summarise_flags

from .conftest import synthetic_tables


def test_clean_synthetic_night_is_reliable(night):
    q = quality_flags(night)
    assert q["reliable"].all()
    assert not q[FLAGS].any().any()


def test_hr_gap_flags_only_affected_epochs():
    labels, hr, acc = synthetic_tables(n_epochs=10)
    hr = hr[(hr["t"] < 120) | (hr["t"] >= 180)]  # epochs 4 and 5 lose all HR
    q = quality_flags(build_night("s", labels, hr, acc))
    assert np.flatnonzero(q["hr_sparse"]).tolist() == [4, 5]
    assert np.flatnonzero(~q["reliable"]).tolist() == [4, 5]


def test_sparse_hr_threshold():
    labels, hr, acc = synthetic_tables(n_epochs=4)
    hr = hr[~((hr["t"] >= 30) & (hr["t"] < 60))]
    hr = pd.concat([hr, pd.DataFrame({"t": [40.0], "hr": [60.0]})]).sort_values("t")
    q = quality_flags(build_night("s", labels, hr, acc), QualityConfig(min_hr_samples=2))
    assert q["hr_sparse"].tolist() == [False, True, False, False]
    q1 = quality_flags(build_night("s", labels, hr, acc), QualityConfig(min_hr_samples=1))
    assert not q1["hr_sparse"].any()


def test_implausible_and_jump_flags():
    labels, hr, acc = synthetic_tables(n_epochs=6)
    hr = hr.copy()
    hr.loc[hr["t"].between(30, 59), "hr"] = 60.0
    hr.loc[hr["t"] == 35.5, "hr"] = 250.0  # implausible (and a jump into epoch 1, back out)
    hr.loc[hr["t"] >= 120, "hr"] = 60.0
    hr.loc[hr["t"] == 125.5, "hr"] = 100.0  # +40 bpm within 5 s -> jump in epoch 4
    q = quality_flags(build_night("s", labels, hr, acc))
    assert np.flatnonzero(q["hr_implausible"]).tolist() == [1]
    assert 4 in np.flatnonzero(q["hr_jump"]) and 1 in np.flatnonzero(q["hr_jump"])
    assert not q.loc[[0, 2, 3, 5], "hr_jump"].any()


def test_jump_after_long_gap_is_not_a_jump():
    labels, hr, acc = synthetic_tables(n_epochs=4)
    hr = pd.DataFrame({"t": [1.0, 6.0, 11.0, 61.0, 66.0], "hr": [60, 60, 60, 100, 100]})
    q = quality_flags(build_night("s", labels, hr, acc))
    assert not q["hr_jump"].any()  # 50 s apart > jump_window_s


def test_accelerometer_gap_flag():
    labels, hr, acc = synthetic_tables(n_epochs=6, acc_hz=10)
    acc = acc[~((acc["t"] >= 60) & (acc["t"] < 80))]  # epoch 2 keeps 1/3 of samples
    q = quality_flags(build_night("s", labels, hr, acc))
    assert np.flatnonzero(q["acc_gap"]).tolist() == [2]


def test_summarise_flags_restricts_to_scored(night):
    q = quality_flags(night)
    q.loc[0, ["hr_sparse", "reliable"]] = [True, False]
    scored = pd.Series([False] + [True] * (len(q) - 1))
    assert summarise_flags(q, scored)["pct_unreliable"] == 0.0
    assert summarise_flags(q)["pct_unreliable"] > 0
