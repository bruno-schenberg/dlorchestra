"""Load a completed experiment's standardized prediction output."""

from pathlib import Path
from typing import Callable

import pandas as pd


def load_fold_predictions(
    experiment_dir: Path,
    id_fn: Callable[[str], str] | None = None,
) -> pd.DataFrame:
    """Concatenate predictions.csv across all fold_* dirs of an experiment.

    Each row gains a "fold" column (the fold directory name). When
    ``id_fn`` is given, an "id" column is added by applying it to the
    "image" column (see :mod:`dlorchestra.analysis.ids` for a regex-based
    factory).

    Raises FileNotFoundError if no fold has a predictions.csv.
    """
    experiment_dir = Path(experiment_dir)
    fold_dirs = sorted(experiment_dir.glob("fold_*"), key=lambda p: int(p.name.split("_")[1]))
    dfs = []
    for fold_dir in fold_dirs:
        pred_file = fold_dir / "predictions.csv"
        if pred_file.exists():
            df = pd.read_csv(pred_file)
            df["fold"] = fold_dir.name
            dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No predictions.csv found under {experiment_dir}/fold_*/")

    combined = pd.concat(dfs, ignore_index=True)
    if id_fn is not None:
        combined["id"] = combined["image"].apply(id_fn)
    return combined
