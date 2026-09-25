# dlorchestra

DL experiment orchestration library. Manages the boilerplate around training deep learning models so you can focus on the model and data.

**Orchestration only** — does not ship model architectures or task-specific training loops. You bring your own `nn.Module` and dataloaders; dlorchestra handles everything else.

## What it does

- **Config-driven experiment runner** — JSON config → training loop, with validation and typo detection
- **Model registry** — generic `Registry[T]` for name-based model construction
- **K-fold CV** — stratified splits with optional held-out test set and group-based stratification
- **Training loop** — AMP, early stopping, learning rate scheduling, checkpoint management
- **Evaluation** — metrics (F2, AUC, precision, recall, specificity), confusion matrices, ROC curves
- **Artifact management** — structured output tree, completion markers for resumability
- **Analysis** — cross-model comparison, fold-level prediction aggregation
- **Subprocess experiments** — run non-PyTorch frameworks (TensorFlow, etc.) via `conda run`

## Install

```bash
pip install dlorchestra
```

Or from source:

```bash
git clone https://github.com/bruno-schenberg/dlorchestra.git
cd dlorchestra
pip install -e .
```

## Quick example

```python
import torch

from dlorchestra.experiment import TorchExperiment
from dlorchestra.registry import Registry

# Register your models — the builder is called as build(name, **kwargs)
models = Registry[torch.nn.Module]()
models.register("my_model", lambda num_classes: MyModel(num_classes=num_classes))

# Subclass TorchExperiment with your data/model wiring
class MyExperiment(TorchExperiment):
    def get_data_list(self, config):
        ...  # -> list of {"path": ..., "label": ...} records

    def build_model(self, config, num_classes):
        return models.build(config["model"], num_classes=num_classes)

    def build_dataloaders(self, train_files, val_files, test_files, config):
        ...  # -> (train_loader, val_loader, test_loader)

# config must supply the keys TorchExperiment reads (name, RANDOM_SEED,
# N_SPLITS, CLASSES, ...) — see `config.validate_config` for typo/enum checks
exp = MyExperiment(experiments_dir="experiments/", device="cuda")
exp.run(config)
```

## Modules

| Module | Purpose |
|--------|---------|
| `experiment` | `BaseExperiment` ABC + `TorchExperiment` (k-fold runner) |
| `registry` | Generic `Registry[T]` for name → builder mappings |
| `config` | Config validation (required keys, enums, typo detection) |
| `training` | Train/val epochs, optimizer/scheduler builders, FocalLoss |
| `evaluate` | Metrics, plots (confusion matrix, ROC), artifact saving |
| `splitting` | K-fold splits with stratification + class weights |
| `seeding` | Reproducible seeding (PyTorch + optional MONAI) |
| `protocols` | `BaseModel` / `BaseDataLoader` structural protocols |
| `subprocess_experiment` | `conda run`-based experiment runner for non-PyTorch |
| `analysis` | Prediction aggregation, regex-based ID extraction |

## Requirements

Python ≥ 3.10. Key dependencies: torch, numpy, pandas, scikit-learn, matplotlib. Optional: monai (for MONAI RNG seeding).

## License

MIT
