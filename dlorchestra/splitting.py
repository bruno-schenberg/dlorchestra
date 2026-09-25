"""Stratified k-fold CV with a permanent held-out test set."""

from collections import Counter
from typing import Callable, Generator

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split


def _stratification_keys(
    data: list[dict],
    group_fn: Callable[[dict], str] | None,
    min_group_size: int,
) -> list[str]:
    """Per-record stratification keys: class label, or "{label}_{group}".

    Groups smaller than min_group_size are folded into "{label}_other" so
    rare groups can't crash StratifiedKFold / train_test_split (which
    require every group to meet the fold/split count).
    """
    labels = [str(d["label"]) for d in data]
    if group_fn is None:
        return labels

    groups = [group_fn(d) for d in data]
    keys = [f"{lab}_{grp}" for lab, grp in zip(labels, groups)]
    counts = Counter(keys)
    return [
        key if counts[key] >= min_group_size else f"{lab}_other"
        for key, lab in zip(keys, labels)
    ]


def split_data(
    all_data: list[dict],
    n_splits: int = 5,
    test_size: float = 0.15,
    seed: int = 42,
    use_kfold: bool = True,
    val_split_ratio: float = 0.20,
    group_fn: Callable[[dict], str] | None = None,
) -> Generator[tuple[int, list[dict], list[dict], list[dict]], None, None]:
    """Stratified split with a permanent held-out test set.

    Each record must have a "label" key. When ``group_fn`` is given,
    stratification uses a composite "{label}_{group_fn(record)}" key instead
    of just the class label, so each fold gets proportional representation
    of the groups (e.g. data source, site, acquisition batch).

    Yields (fold_idx, train_files, val_files, test_files) tuples.
    The test set is identical across all folds.
    """
    min_group_size = n_splits if use_kfold else 2
    keys = _stratification_keys(all_data, group_fn, min_group_size)

    # -- carve out test set first (before any fold logic) --
    can_stratify = all(keys.count(k) >= 2 for k in set(keys))
    dev_data, test_data, dev_keys, _ = train_test_split(
        all_data,
        keys,
        test_size=test_size,
        random_state=seed,
        stratify=keys if can_stratify else None,
    )

    if not use_kfold:
        # single train/val split
        can_stratify_dev = all(dev_keys.count(k) >= 2 for k in set(dev_keys))
        train_data, val_data = train_test_split(
            dev_data,
            test_size=val_split_ratio,
            random_state=seed,
            stratify=dev_keys if can_stratify_dev else None,
        )
        yield (0, train_data, val_data, test_data)
        return

    # k-fold CV on the dev set
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(dev_data, dev_keys)):
        train_files = [dev_data[i] for i in train_idx]
        val_files = [dev_data[i] for i in val_idx]
        yield (fold_idx, train_files, val_files, test_data)


def get_class_weights(data: list[dict], num_classes: int = 2) -> torch.Tensor | None:
    """Inverse-frequency class weights for CrossEntropyLoss.

    Computed from the training fold only to prevent label leakage.
    """
    if not data:
        return None
    labels = [d["label"] for d in data]
    counts = np.bincount(labels, minlength=num_classes)
    total = len(labels)
    weights = np.where(counts > 0, total / counts, 0.0)
    return torch.tensor(weights, dtype=torch.float32)
