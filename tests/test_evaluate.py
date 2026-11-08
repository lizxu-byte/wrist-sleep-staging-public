import numpy as np
import pandas as pd
import pytest

from sleepstage.evaluate import (
    bland_altman,
    evaluate_predictions,
    impute_nearest,
    loso_predict,
    loso_splits,
    night_level_table,
    night_metrics,
    per_subject_kappa,
)
from sleepstage.models import make_model, task_labels


def test_loso_splits_never_mix_participants():
    groups = np.repeat(["a", "b", "c", "d"], [3, 5, 2, 4])
    seen = []
    for g, tr, te in loso_splits(groups):
        assert set(groups[te]) == {g}
        assert g not in set(groups[tr])
        assert len(tr) + len(te) == len(groups)
        seen.append(g)
    assert sorted(seen) == ["a", "b", "c", "d"]


class _Recorder:
    """Dummy model recording which rows it was trained on (rows carry their group id in X)."""

    trained_on: list = []

    def __init__(self, name):
        pass

    def fit(self, X, y):
        _Recorder.trained_on.append(set(X[:, 0].astype(int)))
        self.c = int(np.bincount(y).argmax())
        return self

    def predict(self, X):
        return np.full(len(X), self.c)


def test_loso_predict_holds_out_subject_and_respects_masks():
    groups = np.repeat([0, 1, 2], 10)
    X = groups[:, None].astype(float)
    y = np.tile([0, 1], 15)
    y[0] = -1  # unscored
    train_mask = np.ones(30, bool)
    train_mask[10:15] = False
    _Recorder.trained_on = []
    pred = loso_predict(X, y, groups, "dummy", train_mask=train_mask, model_factory=_Recorder)
    for held_out, used in zip([0, 1, 2], _Recorder.trained_on):
        assert held_out not in used
    assert (pred >= 0).all()  # every row, incl. unscored, gets a prediction


def test_loso_predict_real_models_run():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(4), 50)
    y = rng.integers(0, 2, 200)
    X = np.c_[y + rng.normal(0, 0.3, 200), rng.normal(size=200)]
    for m in ["logreg", "rf"]:
        pred = loso_predict(X, y, groups, m, model_factory=lambda n: make_model(n, n_jobs=1))
        assert (pred == y).mean() > 0.8


def test_task_labels():
    st = np.array([-1, 0, 1, 2, 3, 5])
    assert task_labels(st, "sleep_wake").tolist() == [-1, 0, 1, 1, 1, 1]
    assert task_labels(st, "wake_nrem_rem").tolist() == [-1, 0, 1, 1, 1, 2]
    assert task_labels(st, "wake_light_deep_rem").tolist() == [-1, 0, 1, 1, 2, 3]


def test_night_metrics_known_sequence():
    s = np.array([0, 0, 1, 1, 0, 1, 0, 0])  # 8 epochs = 4 min
    m = night_metrics(s)
    assert m["TIB"] == 4.0 and m["TST"] == 1.5
    assert m["SE"] == pytest.approx(37.5)
    assert m["WASO"] == 0.5 and m["SOL"] == 1.0


def test_night_metrics_valid_mask_and_all_wake():
    s = np.array([1, 0, 1, 1])
    m = night_metrics(s, valid=np.array([True, False, True, True]))
    assert m["TIB"] == 1.5 and m["WASO"] == 0.0
    assert night_metrics(np.zeros(4))["TST"] == 0.0


def test_impute_nearest():
    labels = np.array([1, 9, 9, 0, 9, 9, 9, 1, 9])
    rel = labels != 9
    assert impute_nearest(labels, rel).tolist() == [1, 1, 0, 0, 0, 0, 1, 1, 1]
    assert impute_nearest(labels, np.zeros(9, bool)).tolist() == labels.tolist()


def test_bland_altman():
    r = bland_altman([2, 4, 6], [1, 2, 3])
    assert r["bias"] == pytest.approx(2.0)
    assert r["sd"] == pytest.approx(1.0)
    assert r["loa_low"] == pytest.approx(2 - 1.96) and r["loa_high"] == pytest.approx(2 + 1.96)


def test_per_subject_kappa_and_evaluate():
    y = np.array([0, 1, 0, 1, 0, 1, 1, 1])
    p = np.array([0, 1, 0, 1, 1, 0, 1, 1])
    g = np.array(["a"] * 4 + ["b"] * 4)
    k = per_subject_kappa(y, p, g, 2)
    assert k["a"] == pytest.approx(1.0)
    res = evaluate_predictions(y, p, g, "sleep_wake", np.ones(8, bool))
    assert res["n_epochs"] == 8 and res["accuracy"] == pytest.approx(0.75)
    assert res["confusion"].sum() == 8
    assert res["wake_specificity"] == pytest.approx(2 / 3)
    sub = evaluate_predictions(y, p, g, "sleep_wake", g == "a")
    assert sub["kappa_pooled"] == pytest.approx(1.0) and sub["n_epochs"] == 4


def test_night_level_table_strategies():
    df = pd.DataFrame(
        {"sid": ["a"] * 6, "stage": [-1, 0, 2, 2, 2, 0], "reliable": [True, True, True, False, True, True]}
    )
    pa = np.array([0, 0, 1, 1, 1, 0])
    pg = np.array([0, 0, 1, 0, 1, 0])
    t = night_level_table(df, pa, pg).iloc[0]
    assert t["TST_psg"] == 1.5 and t["TIB_psg"] == 2.5
    assert t["TST_no_gate"] == 1.5
    assert t["TST_impute"] == 1.5  # unreliable epoch takes the earlier neighbour (sleep)
    assert t["TST_exclude"] == 1.0 and t["TIB_exclude"] == 2.0
    assert t["pct_unreliable"] == pytest.approx(20.0)
