"""
Evaluation & Visualization Module for Social Media Sentiment Analysis
Implements Phase D: Metric calculation, Confusion Matrix Heatmaps, and Model Benchmark Plots.
"""

import os
import json
from typing import Dict, Any, List
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend suitable for headless server
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def evaluate_models(
    classifiers: Dict[str, Any],
    X_test,
    y_test,
    class_labels: List[str] = ["negative", "neutral", "positive"],
    save_dir: str = "artifacts"
) -> Dict[str, Any]:
    """
    Evaluates all classifiers on test data.
    Ensures zero_division=0 as mandated by verification checklist.
    Generates comparison metrics and saves plots to disk.
    """
    os.makedirs(save_dir, exist_ok=True)
    results = {}
    f1_scores = {}
    accuracy_scores = {}
    all_confusion_matrices = {}

    for name, clf in classifiers.items():
        y_pred = clf.predict(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
        rec = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

        cm = confusion_matrix(y_test, y_pred, labels=class_labels)
        report_dict = classification_report(
            y_test,
            y_pred,
            labels=class_labels,
            output_dict=True,
            zero_division=0
        )

        results[name] = {
            "accuracy": round(acc, 4),
            "precision_macro": round(prec, 4),
            "recall_macro": round(rec, 4),
            "f1_macro": round(f1, 4),
            "confusion_matrix": cm.tolist(),
            "classification_report": report_dict
        }
        f1_scores[name] = f1
        accuracy_scores[name] = acc
        all_confusion_matrices[name] = cm

    # 1. Generate Confusion Matrices 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (name, cm) in enumerate(all_confusion_matrices.items()):
        ax = axes[idx]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_labels,
            yticklabels=class_labels,
            ax=ax,
            cbar=False
        )
        ax.set_title(f"{name}\n(Macro F1: {f1_scores[name]:.3f})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)

    plt.tight_layout()
    cm_path = os.path.join(save_dir, "confusion_matrices.png")
    fig.savefig(cm_path, dpi=300)
    plt.close(fig)

    # 2. Generate Model Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    model_names = list(f1_scores.keys())
    x = np.arange(len(model_names))
    width = 0.35

    acc_vals = [accuracy_scores[m] for m in model_names]
    f1_vals = [f1_scores[m] for m in model_names]

    rects1 = ax.bar(x - width/2, acc_vals, width, label="Accuracy", color="#3b82f6", alpha=0.9)
    rects2 = ax.bar(x + width/2, f1_vals, width, label="Macro F1-Score", color="#10b981", alpha=0.9)

    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title("Classical ML Algorithm Comparison (Twitter Sentiment)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=10, rotation=10)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Add score labels on bars
    for bar in rects1:
        h = bar.get_height()
        ax.annotate(f"{h:.2%}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for bar in rects2:
        h = bar.get_height()
        ax.annotate(f"{h:.2%}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    comparison_path = os.path.join(save_dir, "model_comparison.png")
    fig.savefig(comparison_path, dpi=300)
    plt.close(fig)

    # 3. Save JSON metrics
    metrics_path = os.path.join(save_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": results,
            "plots": {
                "confusion_matrices": cm_path,
                "model_comparison": comparison_path
            },
            "class_labels": class_labels
        }, f, indent=2)

    return results
