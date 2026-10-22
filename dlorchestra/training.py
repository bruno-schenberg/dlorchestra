"""
Training loop, validation, optimizer and scheduler construction.

No file I/O — all checkpoint state is returned in-memory. Logging only
through the ``logging`` module (no print, no tqdm).
"""

import logging
import time
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import fbeta_score, roc_auc_score
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR

logger = logging.getLogger(__name__)


class EpochTimeoutError(Exception):
    """Raised when a single epoch exceeds the configured time limit."""


class EpochTrainMetrics(NamedTuple):
    loss: float
    acc: float


class ValidationMetrics(NamedTuple):
    loss: float
    acc: float
    f2: float
    auc: float


class DetailedValidationMetrics(NamedTuple):
    loss: float
    acc: float
    f2: float
    auc: float
    details: list[dict]


class TrainingLoopResult(NamedTuple):
    metrics_history: list[dict]
    total_minutes: float
    best_checkpoint: dict | None


# ---------------------------------------------------------------------------
# Optimizer & scheduler
# ---------------------------------------------------------------------------

def build_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=config["LEARNING_RATE"],
        weight_decay=config["WEIGHT_DECAY"],
    )


def build_scheduler(optimizer: torch.optim.Optimizer, config: dict):
    """Returns a per-epoch scheduler, or None if disabled.

    Optional ``WARMUP_EPOCHS`` (default 0) ramps the LR linearly from 1% to
    100% over that many epochs before the configured schedule starts
    (with LR_SCHEDULER "none", the LR then stays constant).
    """
    name = config["LR_SCHEDULER"].lower()
    if name == "none":
        main = None
    elif name == "cosine":
        main = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config["COSINE_T0"],
            T_mult=config["COSINE_T_MULT"],
            eta_min=config["COSINE_ETA_MIN"],
        )
    else:
        raise ValueError(f"Unknown LR_SCHEDULER: '{config['LR_SCHEDULER']}'")

    warmup = config.get("WARMUP_EPOCHS", 0)
    if not warmup:
        return main
    ramp = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup)
    if main is None:
        return ramp
    return SequentialLR(optimizer, [ramp, main], milestones=[warmup])


def resolve_amp_dtype(config: dict) -> torch.dtype:
    """AMP dtype from config["AMP_DTYPE"] — float16 requires GradScaler,
    bfloat16 does not. Defaults to bfloat16."""
    amp_dtype_name = config.get("AMP_DTYPE", "bfloat16")
    return torch.float16 if amp_dtype_name == "float16" else torch.bfloat16


# ---------------------------------------------------------------------------
# Single-epoch passes
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    grad_accum_steps: int = 1,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
    scaler: torch.amp.GradScaler | None = None,
    epoch_timeout: int = 0,
) -> EpochTrainMetrics:
    """One training pass. Returns (epoch_loss, epoch_acc)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    use_amp = use_amp and device != "cpu"
    t_epoch_start = time.time()

    optimizer.zero_grad()
    for step, batch in enumerate(dataloader):
        if epoch_timeout > 0 and (time.time() - t_epoch_start) > epoch_timeout:
            elapsed = time.time() - t_epoch_start
            raise EpochTimeoutError(
                f"Epoch exceeded {epoch_timeout}s timeout at batch {step + 1}/"
                f"{len(dataloader)} ({elapsed:.0f}s elapsed)"
            )
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            if "tabular" in batch:
                outputs = model(images, batch["tabular"].to(device))
            else:
                outputs = model(images)
            loss = criterion(outputs, labels) / grad_accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item() * grad_accum_steps * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    return EpochTrainMetrics(epoch_loss, epoch_acc)


def validate_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: str,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
    return_details: bool = False,
) -> ValidationMetrics | DetailedValidationMetrics:
    """One validation pass.

    Returns (loss, acc, f2, auc) or (loss, acc, f2, auc, details).
    details is a list of per-sample dicts with keys:
      image, true_label, pred_label, prob

    Assumes binary classification: "prob" is softmax[:, 1] (P(class=1)), and
    f2/auc use pos_label=1 throughout.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels: list[int] = []
    all_preds: list[int] = []
    all_probs: list[float] = []
    details: list[dict] = []
    use_amp = use_amp and device != "cpu"

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                if "tabular" in batch:
                    outputs = model(images, batch["tabular"].to(device))
                else:
                    outputs = model(images)
                loss = criterion(outputs, labels)

            probs = torch.softmax(outputs.float(), dim=1)[:, 1]
            preds = outputs.argmax(dim=1)

            running_loss += loss.item() * labels.size(0)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

            if return_details:
                paths = batch["path"]
                for i in range(labels.size(0)):
                    details.append({
                        "image": paths[i],
                        "true_label": labels[i].item(),
                        "pred_label": preds[i].item(),
                        "prob": probs[i].item(),
                    })

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0

    # F2 (beta=2, recall-weighted for clinical/high-recall-priority asymmetry)
    try:
        f2 = fbeta_score(all_labels, all_preds, beta=2, pos_label=1, zero_division=0.0)
    except ValueError:
        f2 = 0.0

    # AUC (threshold-independent)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    if return_details:
        return DetailedValidationMetrics(epoch_loss, epoch_acc, f2, auc, details)
    return ValidationMetrics(epoch_loss, epoch_acc, f2, auc)


# ---------------------------------------------------------------------------
# Focal loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Focal loss for class-imbalanced classification.

    FL = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Reduces the relative loss for well-classified examples, focusing
    training on hard negatives and minority-class samples.
    """

    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(inputs, dim=1)
        p = log_p.exp()
        # Gather the log-prob and prob for the true class
        log_p_t = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - p_t) ** self.gamma
        loss = -focal_weight * log_p_t
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            loss = alpha_t * loss
        return loss.mean()


# ---------------------------------------------------------------------------
# Full training loop (one fold)
# ---------------------------------------------------------------------------

def _run_training_epoch(
    epoch: int,
    num_epochs: int,
    model: nn.Module,
    train_loader,
    val_loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: str,
    grad_accum: int,
    use_amp: bool,
    amp_dtype: torch.dtype,
    scaler: torch.amp.GradScaler | None,
    epoch_timeout: int,
    ckpt_metric: str,
    best_val_score: float,
) -> tuple[dict, dict | None, float]:
    """Train + validate one epoch, and do checkpoint bookkeeping.

    Returns (metrics_row, checkpoint_if_improved_else_None, updated_best_val_score).
    Raises EpochTimeoutError (propagated from train_one_epoch) unchanged.
    """
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    t_epoch = time.time()

    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, device, grad_accum,
        use_amp, amp_dtype, scaler, epoch_timeout,
    )
    val_loss, val_acc, val_f2, val_auc = validate_one_epoch(
        model, val_loader, criterion, device, use_amp, amp_dtype,
    )

    if scheduler is not None:
        scheduler.step()

    epoch_sec = time.time() - t_epoch
    lr = optimizer.param_groups[0]["lr"]

    # GPU memory profiling
    if device != "cpu" and torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        logger.info("GPU peak memory after epoch %d: %.0f MB", epoch, peak_mb)

    metrics_row = {
        "epoch": epoch,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_f2": val_f2,
        "val_auc": val_auc,
        "lr": lr,
        "duration_sec": epoch_sec,
    }

    # Checkpoint on metric improvement
    current_score = val_f2 if ckpt_metric == "val_f2" else val_auc
    improved = current_score > best_val_score
    checkpoint = None
    if improved:
        best_val_score = current_score
        checkpoint = {
            "state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
            "best_epoch": epoch,
            "best_val_auc": val_auc,
            "best_val_f2": val_f2,
        }

    marker = "*" if improved else " "
    logger.info(
        "Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  val_acc=%.4f  "
        "val_AUC=%.4f  val_F2=%.4f%s  lr=%.2e  (%.1fs)",
        epoch, num_epochs, train_loss, val_loss, val_acc,
        val_auc, val_f2, marker, lr, epoch_sec,
    )

    return metrics_row, checkpoint, best_val_score


def run_training_loop(
    model: nn.Module,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: str,
    config: dict,
    class_weights: torch.Tensor | None = None,
    head_param_ids: set[int] | None = None,
) -> TrainingLoopResult:
    """Train for multiple epochs, track best checkpoint, early-stop.

    Checkpoint metric is configurable via config["CHECKPOINT_METRIC"]:
    "val_f2" (default) or "val_auc".

    Returns (metrics_history, total_minutes, best_checkpoint).
    The model is restored to best-checkpoint weights before returning.
    """
    num_epochs = config["EPOCHS"]
    patience = config["EARLY_STOPPING_PATIENCE"]
    grad_accum = config["GRAD_ACCUM_STEPS"]
    use_amp = config["USE_AMP"]
    epoch_timeout = config.get("EPOCH_TIMEOUT", 0)
    ckpt_metric = config.get("CHECKPOINT_METRIC", "val_f2")

    amp_dtype = resolve_amp_dtype(config)
    scaler = None
    if use_amp and amp_dtype == torch.float16 and device != "cpu":
        scaler = torch.amp.GradScaler("cuda")
        logger.info("Using float16 AMP with GradScaler")
    elif use_amp:
        logger.info("Using bfloat16 AMP (no GradScaler)")

    logger.info("Checkpoint metric: %s", ckpt_metric)

    # -- backbone freezing --
    freeze_epochs = config.get("FREEZE_BACKBONE_EPOCHS", 0)
    if freeze_epochs > 0 and head_param_ids:
        frozen_count = 0
        for p in model.parameters():
            if id(p) not in head_param_ids:
                p.requires_grad = False
                frozen_count += 1
        logger.info(
            "Backbone frozen for %d epochs (%d params frozen, %d head params trainable)",
            freeze_epochs, frozen_count, len(head_param_ids),
        )

    # Criterion — instantiated here so class_weights land on the right device
    weight = class_weights.to(device) if class_weights is not None else None
    loss_fn = config.get("LOSS_FN", "cross_entropy")
    if loss_fn == "focal":
        gamma = config.get("FOCAL_GAMMA", 2.0)
        criterion = FocalLoss(alpha=weight, gamma=gamma).to(device)
        logger.info("Using FocalLoss (gamma=%.1f, alpha=%s)", gamma, weight)
    else:
        criterion = nn.CrossEntropyLoss(weight=weight)

    best_val_score = 0.0
    best_checkpoint: dict | None = None
    epochs_no_improve = 0
    metrics_history: list[dict] = []

    t_start = time.time()

    for epoch in range(1, num_epochs + 1):
        # Unfreeze backbone after freeze period
        if freeze_epochs > 0 and epoch == freeze_epochs + 1 and head_param_ids:
            for p in model.parameters():
                p.requires_grad = True
            logger.info("Backbone unfrozen at epoch %d", epoch)

        metrics_row, checkpoint, best_val_score = _run_training_epoch(
            epoch, num_epochs, model, train_loader, val_loader, criterion,
            optimizer, scheduler, device, grad_accum, use_amp, amp_dtype, scaler,
            epoch_timeout, ckpt_metric, best_val_score,
        )
        metrics_history.append(metrics_row)

        if checkpoint is not None:
            best_checkpoint = checkpoint
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early stopping
        if patience > 0 and epochs_no_improve >= patience:
            logger.info("Early stopping at epoch %d (patience=%d)", epoch, patience)
            break

    total_minutes = (time.time() - t_start) / 60.0

    if best_checkpoint is None:
        logger.warning("No epoch achieved %s > 0 — no checkpoint saved", ckpt_metric)
    else:
        model.load_state_dict(best_checkpoint["state_dict"])
        logger.info(
            "Restored best weights from epoch %d (%s=%.4f, val_AUC=%.4f, val_F2=%.4f)",
            best_checkpoint["best_epoch"], ckpt_metric, best_val_score,
            best_checkpoint["best_val_auc"], best_checkpoint["best_val_f2"],
        )

    return TrainingLoopResult(metrics_history, total_minutes, best_checkpoint)
