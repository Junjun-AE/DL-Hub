"""SevSeg-YOLO evaluation metrics (M4-M11).

Metric numbering follows Architecture Design v6.2 §13:
    M1-M3: Detection metrics (mAP etc.) — computed by ultralytics validator
    M4:  Score MAE
    M5:  Spearman rank correlation ρ
    M6:  ±1 Tolerance accuracy
    M7:  Low-end misjudge rate (GT ≤ 3, pred > 5)
    M8:  High-end misjudge rate (GT ≥ 7, pred ≤ 5)
    M9:  11×11 confusion matrix
    M10: Segment-wise MAE (bin-wise, γ-adaptive)
    M11: Visualization (scatter, rank curve, CM heatmap) — in visualization.py

All score inputs are on [0, 10] scale.
"""

from __future__ import annotations

import numpy as np


def _spearmanr(x, y):
    """Spearman rank correlation without scipy dependency."""
    n = len(x)
    if n < 2:
        return 0.0, 1.0
    rank_x = np.argsort(np.argsort(x)).astype(float)
    rank_y = np.argsort(np.argsort(y)).astype(float)
    d2 = np.sum((rank_x - rank_y) ** 2)
    rho = 1 - 6 * d2 / (n * (n ** 2 - 1))
    return float(rho), 0.0  # p-value not computed in fallback


def _get_segment_bins(gamma: float = 1.0):
    """Return segment bins adaptive to gamma (design doc v6.2 section 13.4).

    Args:
        gamma: Non-linear quantization coefficient.
            gamma <= 1:  balanced -> bins [0-3, 4-7, 8-10]
            1 < gamma <= 3: moderate skew -> bins [0-3, 3-5, 5-7, 7-10]
            gamma > 3: severe skew -> bins [0-2, 2-4, 4-6, 6-8, 8-10]

    Returns:
        List of (name, lo, hi) tuples.
    """
    if gamma <= 1.0:
        return [("0-3", 0, 3), ("4-7", 4, 7), ("8-10", 8, 10)]
    elif gamma <= 3.0:
        return [("0-3", 0, 3), ("3-5", 3, 5), ("5-7", 5, 7), ("7-10", 7, 10)]
    else:
        return [("0-2", 0, 2), ("2-4", 2, 4), ("4-6", 4, 6), ("6-8", 6, 8), ("8-10", 8, 10)]


def full_score_evaluation(pred_scores, gt_scores, gamma: float = 1.0):
    """Comprehensive score evaluation (M4-M10).

    Args:
        pred_scores: (N,) predicted scores in [0, 10].
        gt_scores: (N,) GT scores in [0, 10].
        gamma: Non-linear quantization coefficient for adaptive segment bins.

    Returns:
        dict with keys: mae, spearman_rho, spearman_pval, tolerance_accuracy,
        low_end_misjudge, high_end_misjudge, confusion_matrix, segment_mae.
    """
    pred_scores = np.asarray(pred_scores, dtype=float)
    gt_scores = np.asarray(gt_scores, dtype=float)

    results = {}

    # M4: MAE
    results["mae"] = float(np.mean(np.abs(pred_scores - gt_scores)))

    # M5: Spearman rank correlation
    try:
        from scipy.stats import spearmanr
        rho, pval = spearmanr(pred_scores, gt_scores)
    except ImportError:
        rho, pval = _spearmanr(pred_scores, gt_scores)
    results["spearman_rho"] = float(rho) if not np.isnan(rho) else 0.0
    results["spearman_pval"] = float(pval) if not np.isnan(pval) else 1.0

    # M6: +/-1 tolerance accuracy
    results["tolerance_accuracy"] = float(np.mean(np.abs(pred_scores - gt_scores) <= 1.0))

    # M7: Low-end misjudge rate (GT <= 3, pred > 5) — per design doc v6.2 section 13.4
    low_mask = gt_scores <= 3.0
    if low_mask.sum() > 0:
        results["low_end_misjudge"] = float(np.mean(pred_scores[low_mask] > 5.0))
    else:
        results["low_end_misjudge"] = float("nan")

    # M8: High-end misjudge rate (GT >= 7, pred <= 5) — per design doc v6.2 section 13.4
    high_mask = gt_scores >= 7.0
    if high_mask.sum() > 0:
        results["high_end_misjudge"] = float(np.mean(pred_scores[high_mask] <= 5.0))
    else:
        results["high_end_misjudge"] = float("nan")

    # M9: 11x11 confusion matrix (rounded to integers 0-10)
    pred_r = np.clip(np.round(pred_scores), 0, 10).astype(int)
    gt_r = np.clip(np.round(gt_scores), 0, 10).astype(int)
    cm = np.zeros((11, 11), dtype=int)
    for p, g in zip(pred_r, gt_r):
        cm[g, p] += 1
    results["confusion_matrix"] = cm

    # M10: Segment-wise MAE (gamma-adaptive bins per design doc section 13.4)
    segment_mae = {}
    bins = _get_segment_bins(gamma)
    for name, lo, hi in bins:
        seg = (gt_scores >= lo) & (gt_scores <= hi)
        n = int(seg.sum())
        mae = float(np.mean(np.abs(pred_scores[seg] - gt_scores[seg]))) if n > 0 else float("nan")
        segment_mae[name] = {"mae": mae, "n": n}
    results["segment_mae"] = segment_mae

    return results


def print_evaluation_report(results):
    """Print a formatted evaluation report."""
    print("=" * 55)
    print("  SevSeg-YOLO Score Evaluation Report")
    print("=" * 55)
    print(f"  M4  MAE:                {results['mae']:.3f}")
    print(f"  M5  Spearman rho:       {results['spearman_rho']:.3f}")
    print(f"  M6  +/-1 Tolerance Acc: {results['tolerance_accuracy']:.1%}")
    print(f"  M7  Low-end misjudge:   {results.get('low_end_misjudge', float('nan')):.1%}")
    print(f"  M8  High-end misjudge:  {results.get('high_end_misjudge', float('nan')):.1%}")
    seg = results.get("segment_mae", {})
    for name, info in seg.items():
        if isinstance(info, dict):
            print(f"  M10 Segment [{name}]:     MAE={info['mae']:.3f}  (n={info['n']})")
    print("=" * 55)
