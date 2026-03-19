"""SevSeg-YOLO visualization and report generation.

All score values are on [0, 10] scale.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


def plot_score_scatter(pred_scores, gt_scores, save_path=None):
    """GT vs Predicted score scatter plot with diagonal and ±1 tolerance bands.

    Args:
        pred_scores: (N,) predicted scores in [0, 10].
        gt_scores: (N,) GT scores in [0, 10].
        save_path: Optional path to save the figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.scatter(gt_scores, pred_scores, alpha=0.3, s=10, c="steelblue")
    ax.plot([0, 10], [0, 10], "r--", label="Perfect")
    ax.plot([0, 10], [1, 11], "g--", alpha=0.5, label="±1 tolerance")
    ax.plot([0, 10], [-1, 9], "g--", alpha=0.5)
    ax.set_xlabel("GT Score")
    ax.set_ylabel("Predicted Score")
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 10.5)
    ax.set_aspect("equal")
    ax.legend()
    mae = np.mean(np.abs(pred_scores - gt_scores))
    ax.set_title(f"Score Prediction (MAE={mae:.2f})")
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_score_distribution(gt_scores, pred_scores=None, save_path=None):
    """Score distribution histograms: GT and optionally predictions.

    Args:
        gt_scores: (N,) GT scores.
        pred_scores: (N,) optional predicted scores.
        save_path: Optional save path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncols = 2 if pred_scores is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))
    if ncols == 1:
        axes = [axes]

    axes[0].hist(gt_scores, bins=20, range=(0, 10), alpha=0.7, color="steelblue", edgecolor="black")
    axes[0].set_xlabel("Score")
    axes[0].set_ylabel("Count")
    axes[0].set_title("GT Score Distribution")

    if pred_scores is not None:
        axes[1].hist(pred_scores, bins=20, range=(0, 10), alpha=0.7, color="coral", edgecolor="black")
        axes[1].set_xlabel("Score")
        axes[1].set_title("Predicted Score Distribution")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_error_by_class(pred_scores, gt_scores, class_ids, class_names=None, save_path=None):
    """Per-class error box plot.

    Args:
        pred_scores: (N,) predictions.
        gt_scores: (N,) ground truth.
        class_ids: (N,) integer class IDs.
        class_names: Optional dict {id: name}.
        save_path: Optional save path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    errors = np.abs(pred_scores - gt_scores)
    unique_cls = np.unique(class_ids)
    data = []
    labels = []
    for c in unique_cls:
        mask = class_ids == c
        data.append(errors[mask])
        name = class_names.get(int(c), str(int(c))) if class_names else str(int(c))
        labels.append(name)

    fig, ax = plt.subplots(figsize=(max(6, len(unique_cls) * 1.5), 5))
    ax.boxplot(data, labels=labels)
    ax.set_xlabel("Class")
    ax.set_ylabel("Absolute Error")
    ax.set_title("Score Error by Class")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix_11x11(cm, save_path=None):
    """Plot 11×11 score confusion matrix.

    Args:
        cm: (11, 11) integer confusion matrix.
        save_path: Optional save path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_xlabel("Predicted Score")
    ax.set_ylabel("GT Score")
    ax.set_title("Score Confusion Matrix (Rounded to Integer)")
    ax.set_xticks(range(11))
    ax.set_yticks(range(11))
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def find_worst_predictions(pred_scores, gt_scores, image_paths=None, top_k=20):
    """Find top-K worst predictions by absolute error.

    Returns:
        List of dicts with keys: index, pred, gt, error, image (if paths provided).
    """
    errors = np.abs(pred_scores - gt_scores)
    worst_idx = np.argsort(errors)[::-1][:top_k]
    results = []
    for idx in worst_idx:
        entry = {"index": int(idx), "pred": float(pred_scores[idx]),
                 "gt": float(gt_scores[idx]), "error": float(errors[idx])}
        if image_paths is not None:
            entry["image"] = str(image_paths[idx])
        results.append(entry)
    return results


def generate_evaluation_report(pred_scores, gt_scores, save_dir,
                                class_ids=None, class_names=None, image_paths=None):
    """Generate a complete evaluation report with metrics, plots, and worst cases.

    Output structure:
        save_dir/
            metrics.json
            scatter.png
            distribution.png
            confusion_matrix.png
            error_by_class.png  (if class_ids provided)
            worst_cases.json

    Args:
        pred_scores: (N,) predictions in [0, 10].
        gt_scores: (N,) GT in [0, 10].
        save_dir: Output directory.
        class_ids: (N,) optional class IDs.
        class_names: Optional {id: name} dict.
        image_paths: (N,) optional image paths.
    """
    from .evaluation import full_score_evaluation, print_evaluation_report

    os.makedirs(save_dir, exist_ok=True)
    pred_scores = np.asarray(pred_scores, dtype=float)
    gt_scores = np.asarray(gt_scores, dtype=float)

    # 1. Compute metrics
    results = full_score_evaluation(pred_scores, gt_scores)

    # 2. Save metrics JSON (numpy-safe serialization)
    serializable = {}
    for k, v in results.items():
        if isinstance(v, np.ndarray):
            serializable[k] = v.tolist()
        elif isinstance(v, dict):
            serializable[k] = {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv) for kk, vv in v.items()}
        elif isinstance(v, (np.floating, np.integer)):
            serializable[k] = float(v)
        else:
            serializable[k] = v
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(serializable, f, indent=2)

    # 3. Plots
    try:
        plot_score_scatter(pred_scores, gt_scores, os.path.join(save_dir, "scatter.png"))
        plot_score_distribution(gt_scores, pred_scores, os.path.join(save_dir, "distribution.png"))
        if results.get("confusion_matrix") is not None:
            plot_confusion_matrix_11x11(results["confusion_matrix"], os.path.join(save_dir, "confusion_matrix.png"))
        if class_ids is not None:
            plot_error_by_class(pred_scores, gt_scores, class_ids, class_names,
                                os.path.join(save_dir, "error_by_class.png"))
    except ImportError:
        pass  # matplotlib not available

    # 4. Worst cases
    worst = find_worst_predictions(pred_scores, gt_scores, image_paths, top_k=20)
    with open(os.path.join(save_dir, "worst_cases.json"), "w") as f:
        json.dump(worst, f, indent=2)

    # 5. Print
    print_evaluation_report(results)

    return results


def plot_training_curves(csv_path, save_path=None):
    """Plot training loss curves: det losses vs score loss on dual Y-axes (GAP-3 / A17).

    Args:
        csv_path: Path to Ultralytics results.csv.
        save_path: Optional save path for the figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import csv

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader]
    if not rows:
        return

    epochs = list(range(1, len(rows) + 1))
    def col(name):
        return [float(r.get(name, r.get(name.strip(), 0))) for r in rows]

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(epochs, col("train/box_loss"), 'b-', alpha=0.7, label='box_loss')
    ax1.plot(epochs, col("train/cls_loss"), 'g-', alpha=0.7, label='cls_loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Det Loss', color='b')

    ax2 = ax1.twinx()
    ax2.plot(epochs, col("train/score_loss"), 'r-', linewidth=2, label='score_loss')
    ax2.set_ylabel('Score Loss', color='r')

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper right')
    ax1.set_title('SevSeg-YOLO Training Curves')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
