"""
Evaluation metrics, plots, and artifact saving.

All functions are pure data-in / file-out — no live model or DataLoader
required. Plots use matplotlib with the Agg backend (no display needed).
"""

import csv
import json
import logging
import os
import sys
from pathlib import Path

import torch
import matplotlib
# Only force the non-interactive backend if pyplot hasn't been configured by
# the caller yet — don't clobber an app's (e.g. notebook) own backend choice.
if "matplotlib.pyplot" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    auc,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .protocols import MetricsDict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def calculate_fold_metrics(detailed_results: list[dict]) -> MetricsDict:
    """Compute classification metrics from per-sample prediction records.

    Returns dict with: AUC, Precision, Recall, Specificity, F2-Score,
    Accuracy (NaN if not computable).

    Assumes binary classification: class 1 is "positive" (pos_label=1
    throughout), matching detailed_results' "prob" field being P(class=1).
    """
    if not detailed_results:
        return {k: float("nan") for k in
                ("AUC", "Precision", "Recall", "Specificity", "F2-Score", "Accuracy")}

    labels = [r["true_label"] for r in detailed_results]
    preds = [r["pred_label"] for r in detailed_results]
    probs = [r["prob"] for r in detailed_results]

    try:
        auc_val = roc_auc_score(labels, probs)
    except ValueError:
        auc_val = float("nan")

    try:
        prec = precision_score(labels, preds, pos_label=1, zero_division=0.0)
        rec = recall_score(labels, preds, pos_label=1, zero_division=0.0)
        f2 = fbeta_score(labels, preds, beta=2, pos_label=1, zero_division=0.0)
    except ValueError:
        prec = rec = f2 = 0.0

    # Specificity = recall of the negative class
    try:
        spec = recall_score(labels, preds, pos_label=0, zero_division=0.0)
    except ValueError:
        spec = 0.0

    acc = sum(1 for l, p in zip(labels, preds) if l == p) / len(labels)

    return {
        "AUC": auc_val,
        "Precision": prec,
        "Recall": rec,
        "Specificity": spec,
        "F2-Score": f2,
        "Accuracy": acc,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    labels: list[int],
    preds: list[int],
    classes: list[str],
    filename: str | os.PathLike,
    normalize: bool = False,
    title: str = "Confusion Matrix",
) -> None:
    cm = confusion_matrix(labels, preds)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.where(row_sums > 0, cm / row_sums, 0.0)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(title)
    tick_marks = range(len(classes))
    ax.set_xticks(tick_marks, classes)
    ax.set_yticks(tick_marks, classes)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = f"{cm[i, j]:.2f}" if normalize else f"{cm[i, j]}"
            ax.text(j, i, val, ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def plot_roc_curve(
    labels: list[int],
    probs: list[float],
    filename: str | os.PathLike,
) -> None:
    """Plot and save a ROC curve from pre-collected labels and probabilities."""
    if len(set(labels)) < 2:
        logger.warning("ROC curve skipped — only one class present in labels")
        return

    fpr, tpr, _ = roc_curve(labels, probs, pos_label=1)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="darkorange", lw=2,
            label=f"ROC curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-fold artifact saving
# ---------------------------------------------------------------------------

def save_fold_artifacts(
    detailed_results: list[dict],
    metrics_history: list[dict],
    best_checkpoint: dict | None,
    fold_dir: Path,
    classes: list[str],
) -> None:
    """Write all fold-level output files. Pure data-in / file-out.

    Output tree under fold_dir:
        metrics.csv
        predictions.csv
        checkpoints/model_best.pth
        checkpoints/metadata.json
        plots/cm_raw.png
        plots/cm_normalized.png
        plots/roc_curve.png
    """
    plots_dir = fold_dir / "plots"
    ckpt_dir = fold_dir / "checkpoints"
    plots_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    labels = [r["true_label"] for r in detailed_results]
    preds = [r["pred_label"] for r in detailed_results]
    probs = [r["prob"] for r in detailed_results]

    # -- predictions.csv --
    pred_path = fold_dir / "predictions.csv"
    with open(pred_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "true_label", "prediction", "prob"])
        writer.writeheader()
        for r in detailed_results:
            writer.writerow({
                "image": r["image"],
                "true_label": r["true_label"],
                "prediction": r["pred_label"],
                "prob": f"{r['prob']:.6f}",
            })

    # -- metrics.csv (per-epoch + final_eval row) --
    metrics_path = fold_dir / "metrics.csv"
    if metrics_history:
        fieldnames = list(metrics_history[0].keys())
        fold_metrics = calculate_fold_metrics(detailed_results)
        final_row = {"epoch": "final_eval", **{k: "" for k in fieldnames if k != "epoch"}}
        for k, v in fold_metrics.items():
            final_row[k] = f"{v:.4f}" if not np.isnan(v) else "NaN"
        # Add metric columns not in epoch rows
        all_fields = fieldnames + [k for k in fold_metrics if k not in fieldnames]
        with open(metrics_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
            writer.writeheader()
            for row in metrics_history:
                formatted = {}
                for k, v in row.items():
                    formatted[k] = f"{v:.4f}" if isinstance(v, float) else v
                writer.writerow(formatted)
            writer.writerow(final_row)

    # -- plots --
    plot_confusion_matrix(labels, preds, classes,
                          plots_dir / "cm_raw.png",
                          normalize=False, title="Confusion Matrix (Raw)")
    plot_confusion_matrix(labels, preds, classes,
                          plots_dir / "cm_normalized.png",
                          normalize=True, title="Confusion Matrix (Normalized)")
    plot_roc_curve(labels, probs, plots_dir / "roc_curve.png")

    # -- checkpoint --
    if best_checkpoint is not None:
        torch.save(best_checkpoint["state_dict"], ckpt_dir / "model_best.pth")
        metadata = {
            "best_epoch": best_checkpoint["best_epoch"],
            "best_val_auc": best_checkpoint["best_val_auc"],
            "best_val_f2": best_checkpoint["best_val_f2"],
        }
        with open(ckpt_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    logger.info("Fold artifacts saved to %s", fold_dir)


# ---------------------------------------------------------------------------
# Experiment-level summaries
# ---------------------------------------------------------------------------

def save_evaluation_summary(
    all_fold_metrics: dict[str, MetricsDict],
    output_path: Path,
) -> None:
    """Write evaluation_summary.csv: one row per fold + Average + Std. Dev."""
    if not all_fold_metrics:
        logger.warning("No fold metrics to summarize")
        return

    metric_keys = ["AUC", "Precision", "Recall", "Specificity", "F2-Score", "Accuracy"]
    fieldnames = ["Fold"] + metric_keys

    rows: list[dict] = []
    for fold_name, metrics in all_fold_metrics.items():
        row = {"Fold": fold_name}
        for k in metric_keys:
            v = metrics.get(k, float("nan"))
            row[k] = f"{v:.4f}" if not np.isnan(v) else "NaN"
        rows.append(row)

    # Average and Std rows
    for stat_name, fn in [("Average", np.nanmean), ("Std. Dev.", np.nanstd)]:
        row = {"Fold": stat_name}
        for k in metric_keys:
            vals = [metrics.get(k, float("nan")) for metrics in all_fold_metrics.values()]
            v = fn(vals)
            row[k] = f"{v:.4f}" if not np.isnan(v) else "NaN"
        rows.append(row)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Evaluation summary saved to %s", output_path)


def save_summary_json(
    all_fold_metrics: dict[str, MetricsDict],
    output_path: Path,
) -> None:
    """Write summary.json — the experiment completion marker.

    Contains mean/std AUC, best fold, and per-fold metric snapshots.
    Written LAST so its presence signals a fully completed experiment.
    """
    auc_vals = [m.get("AUC", float("nan")) for m in all_fold_metrics.values()]
    valid_aucs = [v for v in auc_vals if not np.isnan(v)]

    best_fold = None
    best_auc = -1.0
    for fold_name, metrics in all_fold_metrics.items():
        fold_auc = metrics.get("AUC", float("nan"))
        if not np.isnan(fold_auc) and fold_auc > best_auc:
            best_auc = fold_auc
            best_fold = fold_name

    summary = {
        "mean_auc": float(np.nanmean(valid_aucs)) if valid_aucs else None,
        "std_auc": float(np.nanstd(valid_aucs)) if valid_aucs else None,
        "best_fold": best_fold,
        "best_fold_auc": best_auc if best_fold else None,
        "num_folds": len(all_fold_metrics),
        "per_fold": {
            name: {k: round(v, 4) if not np.isnan(v) else None for k, v in m.items()}
            for name, m in all_fold_metrics.items()
        },
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Summary written to %s (completion marker)", output_path)
