"""Synthetic data helpers (no real data is used in the tests)."""

import numpy as np
import pandas as pd
import pytest

from sleepstage.io import build_night


def synthetic_tables(n_epochs=40, hr_dt=5.0, acc_hz=10.0, seed=0, stages=None):
    rng = np.random.default_rng(seed)
    t_lab = np.arange(n_epochs) * 30.0
    if stages is None:
        stages = rng.choice([0, 1, 2, 3, 5], size=n_epochs)
    labels = pd.DataFrame({"t": t_lab, "stage": stages})
    t_end = n_epochs * 30.0
    t_hr = np.arange(0.0, t_end, hr_dt) + 0.5
    hr = pd.DataFrame({"t": t_hr, "hr": 60 + rng.normal(0, 2, len(t_hr))})
    t_acc = np.arange(0.0, t_end, 1.0 / acc_hz)
    acc = pd.DataFrame(
        {
            "t": t_acc,
            "x": rng.normal(0, 0.01, len(t_acc)),
            "y": rng.normal(0, 0.01, len(t_acc)),
            "z": -1 + rng.normal(0, 0.01, len(t_acc)),
        }
    )
    return labels, hr, acc


@pytest.fixture
def tables():
    return synthetic_tables()


@pytest.fixture
def night(tables):
    return build_night("synthetic", *tables)
