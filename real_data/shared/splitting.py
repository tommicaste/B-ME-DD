"""Subject-grouped train/val/test split shared by every real-data experiment.

Two stages of `GroupShuffleSplit` give 60/20/20 over subjects. Stage 1 is seeded
identically for every graph and model, so for a given (cohort, seed) the TEST
subjects are the same set no matter which configuration produced the row -- the
property that makes paired comparisons valid. Stage 2 carves the val split used for
hyperparameter selection, so no penalty is ever chosen on windows belonging to a
subject the model was fitted on.
"""
import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def grouped_split(groups, seed):
    groups = np.asarray(groups)

    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    trainval_idx, test_idx = next(gss1.split(np.zeros(len(groups)), groups=groups))

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_rel, val_rel = next(gss2.split(
        np.zeros(len(trainval_idx)),
        groups=groups[trainval_idx]
    ))

    train_idx = trainval_idx[train_rel]
    val_idx = trainval_idx[val_rel]

    assert set(groups[train_idx]).isdisjoint(set(groups[val_idx]))
    assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))
    assert set(groups[val_idx]).isdisjoint(set(groups[test_idx]))

    return train_idx, val_idx, test_idx
