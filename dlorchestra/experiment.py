"""
Experiment runner: fold loop, logging setup, output tree, completion markers.

``BaseExperiment`` owns nothing but the completion-marker convention (a
``summary.json`` file's presence == "this experiment finished"). Subclasses
own the actual run flow — ``TorchExperiment`` implements the standard
config-driven k-fold flow for PyTorch models; a subprocess-based flow for
non-PyTorch tools lives in :mod:`dlorchestra.subprocess_experiment`.

``TorchExperiment`` flow:
  run(config)
    ├── setup logging
    ├── validate_config(config)
    ├── set_seed
    ├── get_data_list → (optional external-data split) → split_data
    └── for each fold:
          check_split → build_dataloaders → build_model → run_training_loop
          → validate (val set) → save_fold_artifacts
          → (EVAL_ON_TEST) validate (test set) → test_predictions.csv
        → save_evaluation_summary
        → save_summary_json  (completion marker, written last)

Everything dataset/model-specific is a hook a subclass overrides; everything
else (the fold loop, AMP training, metrics/artifacts, k-fold splitting) is
shared, generic code from :mod:`dlorchestra.training`/``evaluate``/``splitting``.
"""

import json
import logging
import sys
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

import torch

from . import evaluate, seeding, splitting, training
from .protocols import BaseDataLoader, BaseModel, MetricsDict

logger = logging.getLogger(__name__)


def is_experiment_complete(experiments_dir: Path, name: str, base_name: str | None = None) -> bool:
    """Check if an experiment already has a summary.json (completion marker).

    When running under a job scheduler, the experiment name may carry a
    job-id suffix (e.g. 'exp_12345') but a previous run may have used a
    different job id. So also check for any existing directory matching
    the base name (before the suffix) that already has a summary.json.
    """
    if (experiments_dir / name / "summary.json").exists():
        return True

    if base_name and base_name != name:
        for d in experiments_dir.glob(f"{base_name}_*"):
            if d.is_dir() and (d / "summary.json").exists():
                return True

    return False


class BaseExperiment(ABC):
    """Common contract: an experiment runs a config and produces an output tree."""

    def __init__(self, experiments_dir: Path):
        self.experiments_dir = Path(experiments_dir)

    def is_complete(self, name: str, base_name: str | None = None) -> bool:
        return is_experiment_complete(self.experiments_dir, name, base_name)

    @abstractmethod
    def run(self, config: dict) -> None:
        """Run a complete experiment end-to-end for the given config."""


class TorchExperiment(BaseExperiment):
    """Config-driven k-fold CV experiment runner for PyTorch models.

    ``num_classes`` (from ``len(config["CLASSES"])``) is threaded through
    generically, but the metrics/checkpointing this class delegates to
    (:mod:`dlorchestra.training`/``evaluate``) currently assume binary
    classification: class 1 is "positive" for F2/AUC/precision/recall, and
    checkpointing is driven by ``val_f2``/``val_auc``. Multi-class configs
    will run but those metrics won't be meaningful.

    Required hooks (subclass must implement):
        get_data_list(config) -> list[dict]
            Each record needs at least "label" (int) and "path" (str).
        build_model(config, num_classes) -> nn.Module
        build_dataloaders(train_files, val_files, test_files, config)
            -> (train_loader, val_loader, test_loader)

    Optional hooks (default to a no-op / None):
        validate_config(config) -> None
        get_head_param_ids(model, config) -> set[int] | None
        get_stratify_group_fn(config) -> Callable[[dict], str] | None
        log_extra_fold_info(train_files, val_files, test_files, config) -> None
        check_split(fold_idx, train_files, val_files, test_files, config) -> None

    Config keys read here (besides the training ones): ``RANDOM_SEED``,
    optional ``SPLIT_SEED`` (defaults to ``RANDOM_SEED``; set it to vary
    init/augmentation while keeping folds and the test set fixed),
    ``EVAL_ON_TEST``.
    """

    def __init__(self, experiments_dir: Path, device: str = "cuda"):
        super().__init__(experiments_dir)
        self.device = device

    # -- required hooks --------------------------------------------------

    @abstractmethod
    def get_data_list(self, config: dict) -> list[dict]:
        ...

    @abstractmethod
    def build_model(self, config: dict, num_classes: int) -> BaseModel:
        ...

    @abstractmethod
    def build_dataloaders(
        self, train_files: list[dict], val_files: list[dict], test_files: list[dict], config: dict,
    ) -> tuple[BaseDataLoader, BaseDataLoader, BaseDataLoader]:
        ...

    # -- optional hooks ----------------------------------------------------

    def validate_config(self, config: dict) -> None:
        pass

    def get_head_param_ids(self, model, config: dict) -> set[int] | None:
        return None

    def get_stratify_group_fn(self, config: dict) -> Callable[[dict], str] | None:
        return None

    def log_extra_fold_info(
        self, train_files: list[dict], val_files: list[dict], test_files: list[dict], config: dict,
    ) -> None:
        pass

    def check_split(
        self, fold_idx: int, train_files: list[dict], val_files: list[dict],
        test_files: list[dict], config: dict,
    ) -> None:
        """Called before each fold trains. Raise to abort the experiment,
        e.g. when the test set differs from a frozen list."""

    def split_external_data(
        self, all_data: list[dict], config: dict,
    ) -> tuple[list[dict], list[dict]]:
        """Optionally split off "external" data: appended to every fold's
        training set but never included in val/test. No-op by default —
        override to opt in (e.g. filter by a path prefix or a data flag).

        Returns (own_data, external_data).
        """
        return all_data, []

    def silence_known_warnings(self) -> None:
        """Suppress noisy warnings. Override to add project-specific filters."""
        warnings.filterwarnings("ignore", message=".*TypedStorage is deprecated.*")

    # -- logging setup -------------------------------------------------------

    def _setup_logging(self, experiment_dir: Path, experiment_name: str) -> logging.FileHandler:
        """Configure root logger: console (INFO) + per-experiment file (DEBUG).

        Returns the file handler so it can be removed after the experiment.
        """
        root = logging.getLogger()
        root.setLevel(logging.DEBUG)

        for h in root.handlers[:]:
            root.removeHandler(h)

        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(console)

        log_path = experiment_dir / f"{experiment_name}.log"
        fh = logging.FileHandler(log_path, mode="w")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s  %(name)s  %(levelname)s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        root.addHandler(fh)

        return fh

    # -- run -----------------------------------------------------------------

    def run(self, config: dict) -> None:
        """Run a complete cross-validation experiment end-to-end."""
        name = config["name"]
        experiment_dir = self.experiments_dir / name
        experiment_dir.mkdir(parents=True, exist_ok=True)

        fh = self._setup_logging(experiment_dir, name)
        self.silence_known_warnings()

        try:
            self._run_inner(config, name, experiment_dir)
        finally:
            logging.getLogger().removeHandler(fh)
            fh.close()

    def _run_inner(self, config: dict, name: str, experiment_dir: Path) -> None:
        self.validate_config(config)

        config_path = experiment_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)
        logger.info("Experiment '%s' — config saved to %s", name, config_path)

        seed = config["RANDOM_SEED"]
        seeding.set_seed(seed, cudnn_benchmark=config["CUDNN_BENCHMARK"])

        all_data = self.get_data_list(config)
        logger.info("Loaded %d samples", len(all_data))

        all_data, external_data = self.split_external_data(all_data, config)

        split_seed = config.get("SPLIT_SEED")
        split_seed = seed if split_seed is None else split_seed
        if split_seed != seed:
            logger.info("Split seed %d (training seed %d)", split_seed, seed)

        group_fn = self.get_stratify_group_fn(config)
        fold_splits = splitting.split_data(
            all_data,
            n_splits=config["N_SPLITS"],
            test_size=config["TEST_SPLIT_RATIO"],
            seed=split_seed,
            use_kfold=config["USE_KFOLD"],
            val_split_ratio=config["VAL_SPLIT_RATIO"],
            group_fn=group_fn,
        )

        device = self.device
        all_fold_metrics: dict[str, MetricsDict] = {}
        all_test_metrics: dict[str, MetricsDict] = {}

        for fold_idx, train_files, val_files, test_files in fold_splits:
            self.check_split(fold_idx, train_files, val_files, test_files, config)
            result = self._run_fold(
                fold_idx, train_files, val_files, test_files,
                name=name, experiment_dir=experiment_dir, config=config, device=device,
                external_data=external_data, group_fn=group_fn,
            )
            if result is None:  # epoch timeout — abort the whole experiment
                return
            fold_name, fold_metrics, test_metrics = result
            all_fold_metrics[fold_name] = fold_metrics
            if test_metrics is not None:
                all_test_metrics[fold_name] = test_metrics

        evaluate.save_evaluation_summary(
            all_fold_metrics,
            experiment_dir / "evaluation_summary.csv",
        )
        evaluate.save_summary_json(
            all_fold_metrics,
            experiment_dir / "summary.json",
            test_fold_metrics=all_test_metrics or None,
        )

        logger.info("Experiment '%s' complete", name)

    def _run_fold(
        self,
        fold_idx: int,
        train_files: list[dict],
        val_files: list[dict],
        test_files: list[dict],
        *,
        name: str,
        experiment_dir: Path,
        config: dict,
        device: str,
        external_data: list[dict],
        group_fn: Callable[[dict], str] | None,
    ) -> tuple[str, MetricsDict, MetricsDict | None] | None:
        """Run one fold end-to-end: dataloaders, model, training, eval, artifacts.

        The best checkpoint is always evaluated on val (predictions.csv, fold
        metrics). With EVAL_ON_TEST it is also evaluated on the held-out test
        set (test_predictions.csv, test metrics).

        Returns (fold_name, val_metrics, test_metrics or None), or None if the
        fold hit its epoch timeout — the caller should then abort the whole
        experiment.
        """
        fold_num = fold_idx + 1
        fold_name = f"fold_{fold_num}"
        fold_dir = experiment_dir / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)

        if external_data:
            train_files = train_files + external_data

        logger.info(
            "--- %s %s (train=%d%s, val=%d, test=%d) ---",
            name, fold_name, len(train_files),
            f" [{len(external_data)} external]" if external_data else "",
            len(val_files), len(test_files),
        )

        if group_fn is not None:
            self.log_extra_fold_info(train_files, val_files, test_files, config)

        eval_on_test = config["EVAL_ON_TEST"]
        loader_test_files = test_files if eval_on_test else []
        train_loader, val_loader, test_loader = self.build_dataloaders(
            train_files, val_files, loader_test_files, config,
        )

        class_weights = None
        if config["balancing"] == "weighted_cost_function":
            class_weights = splitting.get_class_weights(train_files, num_classes=len(config["CLASSES"]))

        model = self.build_model(config, num_classes=len(config["CLASSES"]))
        model = model.to(device)

        if config.get("TORCH_COMPILE"):
            logger.info("Compiling model with torch.compile...")
            model = torch.compile(model)

        head_param_ids = None
        if config.get("FREEZE_BACKBONE_EPOCHS", 0) > 0:
            head_param_ids = self.get_head_param_ids(model, config)

        optimizer = training.build_optimizer(model, config)
        scheduler = training.build_scheduler(optimizer, config)

        try:
            metrics_history, train_minutes, best_checkpoint = training.run_training_loop(
                model, train_loader, val_loader,
                optimizer, scheduler, device, config,
                class_weights=class_weights,
                head_param_ids=head_param_ids,
            )
        except training.EpochTimeoutError as e:
            logger.info("")
            logger.info("=" * 60)
            logger.info("EPOCH TIMEOUT — SKIPPING EXPERIMENT '%s'", name)
            logger.info("  %s", e)
            logger.info("=" * 60)
            logger.info("")
            return None

        logger.info("%s %s training complete (%.1f min)", name, fold_name, train_minutes)

        criterion = torch.nn.CrossEntropyLoss()
        amp_dtype = training.resolve_amp_dtype(config)

        def evaluate_on(loader, split_name):
            result = training.validate_one_epoch(
                model, loader, criterion, device,
                use_amp=config["USE_AMP"], amp_dtype=amp_dtype, return_details=True,
            )
            loss, acc, f2, auc, details = result
            logger.info(
                "%s %s %s eval — loss=%.4f  acc=%.4f  AUC=%.4f  F2=%.4f",
                name, fold_name, split_name, loss, acc, auc, f2,
            )
            return details

        val_details = evaluate_on(val_loader, "val")
        evaluate.save_fold_artifacts(
            val_details, metrics_history, best_checkpoint,
            fold_dir, config["CLASSES"],
        )
        fold_metrics = evaluate.calculate_fold_metrics(val_details)

        test_metrics = None
        if eval_on_test:
            test_details = evaluate_on(test_loader, "test")
            evaluate.save_predictions_csv(test_details, fold_dir / "test_predictions.csv")
            test_metrics = evaluate.calculate_fold_metrics(test_details)

        del model, optimizer, scheduler
        del train_loader, val_loader, test_loader
        torch.cuda.empty_cache()

        return fold_name, fold_metrics, test_metrics
