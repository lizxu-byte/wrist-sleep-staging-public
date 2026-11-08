import numpy as np
import pandas as pd
import pytest

from sleepstage.features import (
    build_features,
    feature_columns,
    motion_epoch_metrics,
    rolling,
)
from sleepstage.io import build_night

from .conftest import synthetic_tables


def test_centred_window_is_symmetric():
    x = pd.Series(np.zeros(31))
    x[15] = 1.0
    r = rolling(x, 5, "mean", "centred")
    nonzero = np.flatnonzero(r.to_numpy() > 0)
    assert nonzero.min() == 10 and nonzero.max() == 20
    assert r[15] == pytest.approx(1 / 11)


def test_causal_window_never_uses_future():
    x = pd.Series(np.zeros(31))
    x[15] = 1.0
    r = rolling(x, 5, "mean", "causal")
    nonzero = np.flatnonzero(r.to_numpy() > 0)
    assert nonzero.min() == 15 and nonzero.max() == 25  # only epochs at/after the impulse
    # changing future values does not change earlier causal outputs
    y = x.copy()
    y[20:] = 5.0
    assert np.allclose(rolling(y, 5, "mean", "causal")[:20], r[:20])


def test_window_truncated_at_edges():
    x = pd.Series(np.arange(5, dtype=float))
    r = rolling(x, 2, "mean", "centred")
    assert r[0] == pytest.approx(np.mean([0, 1, 2]))
    assert not r.isna().any()


def test_rolling_rejects_bad_mode():
    with pytest.raises(ValueError):
        rolling(pd.Series([1.0]), 1, "mean", "future")


def _night_with_acc(acc_hz, magnitude):
    labels, hr, _ = synthetic_tables(n_epochs=4)
    t = np.arange(0, 120, 1 / acc_hz)
    acc = pd.DataFrame({"t": t, "x": 0.0, "y": 0.0, "z": magnitude})
    return build_night("s", labels, hr, acc)


def test_stationary_wrist_has_zero_motion():
    m = motion_epoch_metrics(_night_with_acc(50, -1.0))
    assert np.allclose(m["enmo"], 0) and np.allclose(m["mad"], 0) and np.allclose(m["angle"], 0)


def test_enmo_known_value_and_rate_independent():
    m10 = motion_epoch_metrics(_night_with_acc(10, 1.5))
    m50 = motion_epoch_metrics(_night_with_acc(50, 1.5))
    assert np.allclose(m10["enmo"], 500.0) and np.allclose(m50["enmo"], 500.0)
    assert m10["acc_n"].iloc[0] == 300 and m50["acc_n"].iloc[0] == 1500


def test_angle_change_detects_rotation():
    labels, hr, _ = synthetic_tables(n_epochs=2)
    t = np.arange(0, 60, 0.1)
    flat = t < 32.5  # arm flat (z = -1 g) then vertical (x = 1 g)
    acc = pd.DataFrame({"t": t, "x": np.where(flat, 0, 1.0), "y": 0.0, "z": np.where(flat, -1.0, 0)})
    m = motion_epoch_metrics(build_night("s", labels, hr, acc))
    assert m.loc[0, "angle"] == pytest.approx(0.0)
    assert m.loc[1, "angle"] > 0  # the 90 degree change falls in epoch 1


def test_build_features_complete_and_normalised(night):
    df = build_features(night)
    cols = feature_columns()
    assert not df[cols].isna().any().any()
    assert df["hr_z"].median() == pytest.approx(0.0, abs=1e-9)
    assert df["hours"].iloc[0] == 0.0
    assert df["hours"].iloc[-1] == pytest.approx((night.n_epochs - 1) * 30 / 3600)
    assert "hours" not in feature_columns(include_time=False)


def test_features_impute_missing_motion_epochs():
    labels, hr, acc = synthetic_tables(n_epochs=10)
    acc = acc[(acc["t"] < 90) | (acc["t"] >= 150)]  # epochs 3 and 4 have no samples
    df = build_features(build_night("s", labels, hr, acc))
    assert df.loc[[3, 4], "acc_n"].tolist() == [0, 0]
    assert not df[feature_columns()].isna().any().any()
