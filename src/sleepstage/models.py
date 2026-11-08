"""Classifiers and task (label) definitions.

Two scikit-learn models, both with class weighting for the strong class
imbalance (N2 dominates, wake and N3 are rare in this healthy sample):

* ``logreg`` - standardisation + multinomial logistic regression,
  ``class_weight="balanced"``.
* ``rf``     - random forest, 300 trees, ``min_samples_leaf=20`` to limit
  overfitting to individual nights, ``class_weight="balanced_subsample"``.

Gradient boosting was not included: on this feature set it is not expected to
add information over the random forest, and it would add tuning choices that
cannot be validated well with 31 participants.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

#: task -> (class names, mapping from internal stage code to class index)
TASKS: dict[str, tuple[list[str], dict[int, int]]] = {
    "sleep_wake": (["wake", "sleep"], {0: 0, 1: 1, 2: 1, 3: 1, 5: 1}),
    "wake_nrem_rem": (["wake", "NREM", "REM"], {0: 0, 1: 1, 2: 1, 3: 1, 5: 2}),
    "wake_light_deep_rem": (["wake", "light", "deep", "REM"], {0: 0, 1: 1, 2: 1, 3: 2, 5: 3}),
}
TASK_LABELS = {
    "sleep_wake": "(a) sleep vs wake",
    "wake_nrem_rem": "(b) wake / NREM / REM",
    "wake_light_deep_rem": "(c) wake / light / deep / REM",
}
MODELS = ["logreg", "rf"]


def task_labels(stage: np.ndarray, task: str) -> np.ndarray:
    """Map internal stage codes to class indices for ``task``; -1 for unscored epochs."""
    mapping = TASKS[task][1]
    stage = np.asarray(stage)
    y = np.full(stage.shape, -1, dtype=int)
    for code, cls in mapping.items():
        y[stage == code] = cls
    return y


def make_model(name: str, random_state: int = 0, n_jobs: int = -1) -> Pipeline:
    if name == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0),
        )
    if name == "rf":
        return make_pipeline(
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=20,
                max_features="sqrt",
                class_weight="balanced_subsample",
                n_jobs=n_jobs,
                random_state=random_state,
            )
        )
    raise ValueError(f"unknown model {name!r}")
