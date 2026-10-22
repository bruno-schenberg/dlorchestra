# Changelog

All notable changes to this project are documented in this file.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

## [0.2.0] - 2026-10-22

### Changed
- **Breaking (output semantics):** with `EVAL_ON_TEST`, each fold's best
  checkpoint is now evaluated on **both** val and test. `predictions.csv` and
  the fold metrics are always val (previously they held test results when
  `EVAL_ON_TEST` was on), and the test results go to `test_predictions.csv`
  plus a `"test"` block (mean/std AUC, per-fold metrics) in `summary.json`.

### Added
- `SPLIT_SEED` config key (optional, defaults to `RANDOM_SEED`): seeds the
  train/val/test split separately, so seed repeats keep the same folds and test set
- `WARMUP_EPOCHS` config key (optional, default 0): linear LR warmup (1% → 100%)
  before the configured schedule
- `TorchExperiment.check_split` hook, called before each fold trains (e.g. to
  verify the test split against a frozen list)
- `evaluate.save_predictions_csv`

### Fixed
- README's "Quick example" now matches `TorchExperiment`'s real API
- `pyproject.toml` gained `readme`/`license`/`classifiers`/`authors`/
  `[project.urls]` metadata; added a `py.typed` marker (PEP 561)
- Removed the hardcoded `STRATIFY_BY_SCANNER`/`EXTERNAL_DATA_PREFIX` config-key
  coupling from `TorchExperiment._run_inner` — stratification is now gated on
  whether `get_stratify_group_fn` returns a group function, and external-data
  handling moved to a new optional `split_external_data` hook
  (**breaking** for subclasses relying on the old default `EXTERNAL_DATA_PREFIX`
  behavior — override `split_external_data` instead)
- `evaluate.py` only forces the `Agg` matplotlib backend if pyplot hasn't
  already been configured by the caller
- `subprocess_experiment.py` uses `logging` instead of `print()`

### Changed
- Split `experiment.py`'s `_run_inner` into `_run_inner` + `_run_fold`, and
  `training.py`'s `run_training_loop` into `run_training_loop` +
  `_run_training_epoch`
- `protocols.py`: dropped unused `ModelFactory`/`DataListFn`/
  `DataLoaderFactory` aliases; wired `BaseModel`/`BaseDataLoader`/
  `MetricsDict` into real signatures
- `train_one_epoch`/`validate_one_epoch`/`run_training_loop` now return
  `NamedTuple`s instead of plain tuples (still positionally unpackable)
- Extracted `training.resolve_amp_dtype(config)`, deduping AMP-dtype
  selection that was duplicated in `experiment.py` and `training.py`
- `torch.cuda.reset_peak_memory_stats()` now runs each epoch, so the logged
  GPU peak-memory figure is per-epoch instead of a cumulative running peak
- `analysis.predictions.load_fold_predictions` sorts `fold_*` directories
  numerically instead of lexicographically
- Documented the binary-classification assumption (`pos_label=1`,
  `softmax[:, 1]`, F2/AUC) baked into `training.py`/`evaluate.py`
- Dropped the `dev = ["pytest"]` extra — no tests exist yet to back it

## [0.1.0] - Initial release

Extracted from a private aneurysm-detection pipeline: config-driven k-fold
CV experiment runner, AMP training loop, metrics/plotting, and
cross-model analysis helpers.
