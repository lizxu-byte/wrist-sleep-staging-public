import numpy as np
import pandas as pd
import pytest

from sleepstage import io
from sleepstage.io import assign_epochs, build_night, epoch_heart_rate

from .conftest import synthetic_tables


def test_assign_epochs_half_open_intervals():
    t = np.array([-0.1, 0.0, 29.999, 30.0, 59.9, 60.0, 89.9, 90.0])
    assert assign_epochs(t, 0.0, 3).tolist() == [-1, 0, 0, 1, 1, 2, 2, -1]


def test_assign_epochs_offset_start():
    assert assign_epochs(np.array([100.0, 129.0, 130.0]), 100.0, 2).tolist() == [0, 0, 1]


def test_build_night_crops_sorts_and_deduplicates():
    labels, hr, acc = synthetic_tables(n_epochs=4)
    extra = pd.DataFrame({"t": [-500.0, 10.0, 10.0, 5000.0], "hr": [99, 70, 71, 99]})
    hr = pd.concat([hr.iloc[::-1], extra], ignore_index=True)  # reversed + duplicates + outside
    n = build_night("s", labels, hr, acc)
    assert np.all(np.diff(n.hr_t) > 0)
    assert n.hr_t.min() >= 0 and n.hr_t.max() < 120
    assert (n.hr_t == 10.0).sum() == 1


def test_stage_mapping_and_unscored_kept():
    labels, hr, acc = synthetic_tables(n_epochs=6, stages=[-1, 0, 1, 2, 4, 5])
    n = build_night("s", labels, hr, acc)
    assert n.stage.tolist() == [-1, 0, 1, 2, 4, 5]  # mapping happens in read_labels
    df = pd.DataFrame({"t": [0, 30, 60], "stage": [4, -1, 5]})
    assert df["stage"].map(io.STAGE_MAP).tolist() == [3, -1, 5]


def test_read_labels_maps_codes_and_rejects_unknown(tmp_path):
    p = tmp_path / "1_labeled_sleep.txt"
    p.write_text("0 0\n30 4\n60 -1\n90 5\n")
    assert io.read_labels(p)["stage"].tolist() == [0, 3, -1, 5]
    p.write_text("0 0\n30 7\n")
    with pytest.raises(ValueError):
        io.read_labels(p)


def test_irregular_label_grid_raises():
    labels, hr, acc = synthetic_tables(n_epochs=3)
    labels.loc[2, "t"] = 75.0
    with pytest.raises(ValueError):
        build_night("s", labels, hr, acc)


def test_epoch_hr_mean_and_interpolation():
    labels = pd.DataFrame({"t": [0.0, 30.0, 60.0], "stage": [0, 2, 2]})
    hr = pd.DataFrame({"t": [5.0, 15.0, 85.0], "hr": [60.0, 70.0, 80.0]})
    acc = pd.DataFrame({"t": [0.0], "x": [0.0], "y": [0.0], "z": [1.0]})
    e = epoch_heart_rate(build_night("s", labels, hr, acc))
    assert e["hr_n"].tolist() == [2, 0, 1]
    assert e.loc[0, "hr_mean"] == 65.0
    assert np.isnan(e.loc[1, "hr_mean"])
    # epoch 1 midpoint (45 s) interpolated between (15, 70) and (85, 80)
    assert e.loc[1, "hr_interp"] == pytest.approx(70 + 10 * 30 / 70)
    assert np.isnan(e.loc[2, "hr_sd"])  # single sample -> no SD
    assert e.loc[0, "hr_sd"] == pytest.approx(5.0)


def test_list_subjects_requires_all_files(tmp_path):
    for d in ["labels", "heart_rate", "motion"]:
        (tmp_path / d).mkdir()
    (tmp_path / "labels" / "1_labeled_sleep.txt").write_text("0 0\n")
    (tmp_path / "heart_rate" / "1_heartrate.txt").write_text("0,60\n")
    (tmp_path / "motion" / "1_acceleration.txt").write_text("0 0 0 1\n")
    (tmp_path / "labels" / "2_labeled_sleep.txt").write_text("0 0\n")
    assert io.list_subjects(tmp_path) == ["1"]
    n = io.load_night(tmp_path, "1")
    assert n.n_epochs == 1 and n.acc.shape == (1, 3)
