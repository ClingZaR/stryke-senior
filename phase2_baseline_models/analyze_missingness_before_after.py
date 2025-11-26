#!/usr/bin/env python3
"""
Phase 2: Missingness Impact Analysis (Before vs After)
Focus: Explore whether missing features (e.g., HbA1c) are hurting performance and if better imputation fixes it.
This script produces a clean before/after comparison centered on missingness handling only.

Outputs:
- visuals/missingness/roc_before_after.png
- visuals/missingness/pr_before_after.png
- visuals/missingness/confusion_matrices_before_after.png
- visuals/missingness/missingness_rates.png
- reports/missingness/missingness_before_after.md
- reports/missingness/missingness_metrics.json
"""

import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")
plt.style.use("default")

# ---------- Utilities ----------

# Path helpers
BASE_DIR = Path(__file__).parent.resolve()
ROOT_DIR = BASE_DIR.parent.resolve()

def p(rel: str) -> str:
    return str((BASE_DIR / rel).resolve())

def ensure_dirs():
    os.makedirs(p("visuals/missingness"), exist_ok=True)
    os.makedirs(p("reports/missingness"), exist_ok=True)


def split_target(df: pd.DataFrame):
    # Prefer 'stroke' if available, else map 'Death outcome (YES/NO)'
    if "stroke" in df.columns:
        y = df["stroke"].astype(int)
        X = df.drop(columns=["stroke"]).copy()
        target_name = "stroke"
    elif "Death outcome (YES/NO)" in df.columns:
        y = df["Death outcome (YES/NO)"].map({"yes": 1, "no": 0}).fillna(0).astype(int)
        X = df.drop(columns=["Death outcome (YES/NO)"]).copy()
        target_name = "Death outcome (YES/NO)"
    else:
        raise ValueError("Target not found. Expected 'stroke' or 'Death outcome (YES/NO)'.")
    return X, y, target_name


def label_encode_safe(train_col: pd.Series, test_col: pd.Series):
    # Safe label encoding: map unseen categories in test to -1
    train_vals = train_col.fillna("Unknown").astype(str)
    test_vals = test_col.fillna("Unknown").astype(str)
    uniques = pd.Series(train_vals.unique()).reset_index()
    mapping = {v: i for i, v in enumerate(uniques[0].tolist())}
    train_enc = train_vals.map(mapping).astype(int)
    test_enc = test_vals.map(lambda v: mapping.get(v, -1)).astype(int)
    return train_enc, test_enc


def preprocess_before(X_train, X_test):
    # BEFORE: simple median imputation for numeric, safe label encoding for categoricals, standard scaling
    Xtr = X_train.copy()
    Xte = X_test.copy()

    cat_cols = Xtr.select_dtypes(include=["object"]).columns.tolist()
    num_cols = Xtr.select_dtypes(include=[np.number]).columns.tolist()

    # Encode categoricals safely
    for col in cat_cols:
        Xtr[col], Xte[col] = label_encode_safe(Xtr[col], Xte[col])
        if col not in num_cols:
            num_cols.append(col)  # encoded to ints, treat as numeric afterwards

    # Impute numeric
    simp = SimpleImputer(strategy="median")
    Xtr[num_cols] = simp.fit_transform(Xtr[num_cols])
    Xte[num_cols] = simp.transform(Xte[num_cols])

    # Scale
    scaler = StandardScaler()
    Xtr[num_cols] = scaler.fit_transform(Xtr[num_cols])
    Xte[num_cols] = scaler.transform(Xte[num_cols])

    return Xtr, Xte


def preprocess_after(X_train, X_test, original_train, original_test):
    # AFTER: add missingness indicators + KNN imputation for numeric, safe label encoding for categoricals, standard scaling
    Xtr = X_train.copy()
    Xte = X_test.copy()

    # Identify columns with any missing in original (pre-split) views for indicator creation
    miss_cols = [c for c in original_train.columns if original_train[c].isna().any() or original_test[c].isna().any()]

    # Add missingness indicator columns (0/1)
    for col in miss_cols:
        Xtr[f"is_missing__{col}"] = original_train[col].isna().astype(int)
        Xte[f"is_missing__{col}"] = original_test[col].isna().astype(int)

    cat_cols = Xtr.select_dtypes(include=["object"]).columns.tolist()
    num_cols = Xtr.select_dtypes(include=[np.number]).columns.tolist()

    # Encode categoricals safely
    for col in cat_cols:
        Xtr[col], Xte[col] = label_encode_safe(Xtr[col], Xte[col])
        if col not in num_cols:
            num_cols.append(col)

    # KNN impute numeric
    knn = KNNImputer(n_neighbors=5, weights="distance")
    Xtr[num_cols] = knn.fit_transform(Xtr[num_cols])
    Xte[num_cols] = knn.transform(Xte[num_cols])

    # Scale
    scaler = StandardScaler()
    Xtr[num_cols] = scaler.fit_transform(Xtr[num_cols])
    Xte[num_cols] = scaler.transform(Xte[num_cols])

    return Xtr, Xte


def train_eval(Xtr, Xte, ytr, yte, name):
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:, 1]
    # keep default 0.5 for baseline view
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(yte, pred)),
        "f1": float(f1_score(yte, pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(yte, proba)),
        "auprc": float(average_precision_score(yte, proba)),
        "pred": pred,
        "proba": proba,
        "name": name,
        "model": model,
    }
    return model, metrics


def plot_missingness_rates(train_df, test_df, out_path):
    # Compute percent missing per column (train view)
    miss = train_df.isna().mean().sort_values(ascending=False)
    top = miss.head(20)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top.values * 100, y=top.index, color="#3498db")
    plt.xlabel("% Missing (Train)")
    plt.ylabel("Feature")
    plt.title("Top Missingness Rates (Train)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc_pr(results, y_true, roc_path, pr_path):
    # ROC
    plt.figure(figsize=(8, 7))
    for r in results:
        fpr, tpr, _ = roc_curve(y_true, r["proba"]) 
        plt.plot(fpr, tpr, linewidth=2, label=f"{r['name']} (AUC={r['auc_roc']:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC: Before vs After Missingness Handling", fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(roc_path, dpi=300, bbox_inches="tight")
    plt.close()

    # PR
    plt.figure(figsize=(8, 7))
    for r in results:
        precision, recall, _ = precision_recall_curve(y_true, r["proba"]) 
        plt.plot(recall, precision, linewidth=2, label=f"{r['name']} (AUPRC={r['auprc']:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall: Before vs After", fontweight="bold")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(pr_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusions(results, y_true, out_path):
    fig, axes = plt.subplots(1, len(results), figsize=(12, 5))
    if len(results) == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        cm = confusion_matrix(y_true, r["pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
        ax.set_title(f"{r['name']} Confusion", fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def sweep_thresholds(y_true, proba, precision_floor=0.8):
    thresholds = np.linspace(0.0, 1.0, 101)
    f1s, precisions, recalls = [], [], []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        p = precision_score(y_true, pred, zero_division=0)
        r = recall_score(y_true, pred, zero_division=0)
        f = f1_score(y_true, pred, zero_division=0)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
    # best F1 threshold
    best_f1_idx = int(np.argmax(f1s))
    best_f1_threshold = float(thresholds[best_f1_idx])
    best_f1 = float(f1s[best_f1_idx])

    # among thresholds meeting precision >= precision_floor, maximize recall
    valid_idxs = [i for i, p in enumerate(precisions) if p >= precision_floor]
    if valid_idxs:
        best_rec_idx = max(valid_idxs, key=lambda i: recalls[i])
        p80_threshold = float(thresholds[best_rec_idx])
        p80_precision = float(precisions[best_rec_idx])
        p80_recall = float(recalls[best_rec_idx])
    else:
        p80_threshold = None
        p80_precision = None
        p80_recall = None

    return {
        "thresholds": thresholds.tolist(),
        "f1s": f1s,
        "precisions": precisions,
        "recalls": recalls,
        "best_f1_threshold": best_f1_threshold,
        "best_f1": best_f1,
        "p80_threshold": p80_threshold,
        "p80_precision": p80_precision,
        "p80_recall": p80_recall,
    }


def plot_threshold_sweep(y_true, before, after, f1_path, pr_path):
    # before/after are dicts returned by sweep_thresholds
    th = np.array(before["thresholds"])  # same grid used for both

    # F1 sweep
    plt.figure(figsize=(9, 6))
    plt.plot(th, before["f1s"], label="Before - F1", color="#999")
    plt.plot(th, after["f1s"], label="After - F1", color="#2c7fb8")
    if after.get("best_f1_threshold") is not None:
        plt.axvline(after["best_f1_threshold"], color="#2c7fb8", linestyle="--", alpha=0.7,
                    label=f"After best F1 thr={after['best_f1_threshold']:.2f}")
    if before.get("best_f1_threshold") is not None:
        plt.axvline(before["best_f1_threshold"], color="#999", linestyle=":", alpha=0.7,
                    label=f"Before best F1 thr={before['best_f1_threshold']:.2f}")
    plt.xlabel("Threshold")
    plt.ylabel("F1")
    plt.title("Threshold Sweep: F1 vs Threshold", fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f1_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Precision/Recall sweep
    plt.figure(figsize=(9, 6))
    plt.plot(th, before["precisions"], label="Before - Precision", color="#d95f0e")
    plt.plot(th, before["recalls"], label="Before - Recall", color="#fec44f")
    plt.plot(th, after["precisions"], label="After - Precision", color="#1b9e77")
    plt.plot(th, after["recalls"], label="After - Recall", color="#66a61e")
    # mark p>=0.8 chosen points if exist
    if after.get("p80_threshold") is not None:
        plt.axvline(after["p80_threshold"], color="#1b9e77", linestyle="--", alpha=0.6,
                    label=f"After P>=0.8 thr={after['p80_threshold']:.2f}")
    if before.get("p80_threshold") is not None:
        plt.axvline(before["p80_threshold"], color="#d95f0e", linestyle=":", alpha=0.6,
                    label=f"Before P>=0.8 thr={before['p80_threshold']:.2f}")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Threshold Sweep: Precision/Recall vs Threshold", fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(pr_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusions_with_thresholds(results, y_true, thresholds, out_path, title_suffix):
    # results: list of dicts with keys name, proba
    fig, axes = plt.subplots(1, len(results), figsize=(12, 5))
    if len(results) == 1:
        axes = [axes]
    for ax, r, thr in zip(axes, results, thresholds):
        if thr is None:
            ax.axis('off')
            ax.set_title(f"{r['name']} (no thr)")
            continue
        pred = (r["proba"] >= thr).astype(int)
        cm = confusion_matrix(y_true, pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
        ax.set_title(f"{r['name']} @ thr={thr:.2f} {title_suffix}", fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

# New: grid of confusions across multiple fixed thresholds for side-by-side comparisons

def plot_confusion_grid_for_thresholds(results, y_true, thresholds, out_path, suptitle):
    n_rows = len(results)
    n_cols = len(thresholds)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 3.2 * n_rows))
    # Ensure axes is 2D array
    if n_rows == 1:
        axes = np.array([axes])
    for i, res in enumerate(results):
        for j, thr in enumerate(thresholds):
            ax = axes[i, j]
            pred = (res["proba"] >= thr).astype(int)
            cm = confusion_matrix(y_true, pred)
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=ax,
                cbar=False,
                xticklabels=["Neg", "Pos"],
                yticklabels=["Neg", "Pos"],
            )
            if i == 0:
                ax.set_title(f"thr={thr:.1f}")
            if j == 0:
                ax.set_ylabel(f"{res['name']}\nActual")
            else:
                ax.set_ylabel("")
            ax.set_xlabel("Predicted")
    plt.suptitle(suptitle, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_indicator_importance(after_model, after_columns, out_path):
    if not hasattr(after_model, "feature_importances_"):
        return None
    fi = pd.Series(after_model.feature_importances_, index=after_columns)
    fi_ind = fi[fi.index.str.startswith("is_missing__")].sort_values(ascending=False)
    if fi_ind.empty:
        return None
    top = fi_ind.head(15)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top.values, y=top.index, color="#6baed6")
    plt.xlabel("Importance")
    plt.ylabel("Missingness Indicator")
    plt.title("Top Missingness Indicator Importances (After Model)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return fi_ind


def write_report(metrics_before, metrics_after, hba1c_info, report_md_path, report_json_path, thr_info=None, top_indicators=None):
    lines = []
    lines.append("# Missingness Impact: Before vs After (Phase 2)\n")
    lines.append(f"- Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("\n## Summary\n")
    lines.append("Baseline (median, no indicators) vs improved (indicators + KNN).\n")
    lines.append("\n## Metrics (Default 0.5 threshold)\n")
    lines.append(f"- BEFORE — Acc: {metrics_before['accuracy']:.3f}, F1: {metrics_before['f1']:.3f}, AUC: {metrics_before['auc_roc']:.3f}, AUPRC: {metrics_before['auprc']:.3f}\n")
    lines.append(f"- AFTER  — Acc: {metrics_after['accuracy']:.3f}, F1: {metrics_after['f1']:.3f}, AUC: {metrics_after['auc_roc']:.3f}, AUPRC: {metrics_after['auprc']:.3f}\n")

    if thr_info is not None:
        lines.append("\n## Threshold Optimization\n")
        lines.append(f"- BEFORE best-F1 threshold: {thr_info['before']['best_f1_threshold']:.2f} (F1={thr_info['before']['best_f1']:.3f})\n")
        lines.append(f"- AFTER  best-F1 threshold: {thr_info['after']['best_f1_threshold']:.2f} (F1={thr_info['after']['best_f1']:.3f})\n")
        if thr_info['before'].get('p80_threshold') is not None:
            lines.append(f"- BEFORE P>=0.8 threshold: {thr_info['before']['p80_threshold']:.2f} (P={thr_info['before']['p80_precision']:.3f}, R={thr_info['before']['p80_recall']:.3f})\n")
        if thr_info['after'].get('p80_threshold') is not None:
            lines.append(f"- AFTER  P>=0.8 threshold: {thr_info['after']['p80_threshold']:.2f} (P={thr_info['after']['p80_precision']:.3f}, R={thr_info['after']['p80_recall']:.3f})\n")

    if hba1c_info is not None:
        lines.append("\n## HbA1c Missingness\n")
        lines.append(f"- Column: {hba1c_info['col']}\n")
        lines.append(f"- Train missing rate: {hba1c_info['train_missing']*100:.1f}%\n")
        lines.append(f"- Test missing rate: {hba1c_info['test_missing']*100:.1f}%\n")

    if top_indicators is not None and len(top_indicators) > 0:
        lines.append("\n## Top Missingness Indicators (After model)\n")
        for i, (k, v) in enumerate(top_indicators.head(5).items()):
            lines.append(f"{i+1}. {k}: {float(v):.4f}")
        lines.append("\n")

    lines.append("\n## Visuals Generated\n")
    lines.append("- visuals/missingness/missingness_rates.png\n")
    lines.append("- visuals/missingness/roc_before_after.png\n")
    lines.append("- visuals/missingness/pr_before_after.png\n")
    lines.append("- visuals/missingness/confusion_matrices_before_after.png\n")
    lines.append("- visuals/missingness/threshold_sweep_f1.png\n")
    lines.append("- visuals/missingness/threshold_sweep_precision_recall.png\n")
    lines.append("- visuals/missingness/confusion_opt_f1.png\n")
    lines.append("- visuals/missingness/confusion_p80.png\n")
    lines.append("- visuals/missingness/confusion_grid_fixed_thresholds.png\n")
    lines.append("- visuals/missingness/indicator_importance.png\n")

    with open(report_md_path, "w") as f:
        f.write("\n".join(lines))

    out_json = {
        "timestamp": datetime.now().isoformat(),
        "before": {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in metrics_before.items() if k in ["accuracy","f1","auc_roc","auprc"]},
        "after": {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in metrics_after.items() if k in ["accuracy","f1","auc_roc","auprc"]},
        "hba1c": hba1c_info,
        "thresholds": thr_info,
        "top_indicator_importances": top_indicators.head(10).to_dict() if top_indicators is not None else None,
    }
    with open(report_json_path, "w") as f:
        json.dump(out_json, f, indent=2)


def main():
    ensure_dirs()

    # Load dataset from project root
    df = pd.read_csv(str((ROOT_DIR / "clean_data.csv").resolve()))

    # Drop obvious non-feature IDs if present
    drop_cols = ["CMRN", "Unnamed: 0"]
    present_drop = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=present_drop)

    X, y, target_name = split_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    plot_missingness_rates(X_train, X_test, p("visuals/missingness/missingness_rates.png"))

    Xtr_before, Xte_before = preprocess_before(X_train, X_test)
    Xtr_after, Xte_after = preprocess_after(X_train, X_test, X_train, X_test)

    model_b, m_before = train_eval(Xtr_before, Xte_before, y_train, y_test, name="Before (Median)")
    model_a, m_after = train_eval(Xtr_after, Xte_after, y_train, y_test, name="After (Indicators+KNN)")

    plot_roc_pr([m_before, m_after], y_test,
                p("visuals/missingness/roc_before_after.png"),
                p("visuals/missingness/pr_before_after.png"))

    plot_confusions([m_before, m_after], y_test,
                    p("visuals/missingness/confusion_matrices_before_after.png"))

    before_thr = sweep_thresholds(y_test, m_before["proba"], precision_floor=0.8)
    after_thr = sweep_thresholds(y_test, m_after["proba"], precision_floor=0.8)

    plot_threshold_sweep(y_test, before_thr, after_thr,
                         p("visuals/missingness/threshold_sweep_f1.png"),
                         p("visuals/missingness/threshold_sweep_precision_recall.png"))

    plot_confusions_with_thresholds(
        [m_before, m_after], y_test,
        thresholds=[before_thr["best_f1_threshold"], after_thr["best_f1_threshold"]],
        out_path=p("visuals/missingness/confusion_opt_f1.png"),
        title_suffix="(Best F1)"
    )

    plot_confusions_with_thresholds(
        [m_before, m_after], y_test,
        thresholds=[before_thr.get("p80_threshold"), after_thr.get("p80_threshold")],
        out_path=p("visuals/missingness/confusion_p80.png"),
        title_suffix="(P>=0.8)"
    )

    # New: Grid across fixed thresholds 0.1..0.9
    fixed_thresholds = [round(t, 1) for t in np.arange(0.1, 1.0, 0.1)]
    plot_confusion_grid_for_thresholds(
        [m_before, m_after], y_test, fixed_thresholds,
        out_path=p("visuals/missingness/confusion_grid_fixed_thresholds.png"),
        suptitle="Confusion matrices across fixed thresholds"
    )

    # Indicator importance snapshot (After model)
    top_ind = plot_indicator_importance(
        m_after["model"], Xtr_after.columns.tolist(),
        out_path=p("visuals/missingness/indicator_importance.png")
    )

    # HbA1c check (if present)
    hba1c_col = None
    for c in X.columns:
        if "hba1c" in c.lower():
            hba1c_col = c
            break
    hba1c_info = None
    if hba1c_col is not None:
        hba1c_info = {
            "col": hba1c_col,
            "train_missing": float(X_train[hba1c_col].isna().mean()),
            "test_missing": float(X_test[hba1c_col].isna().mean()),
        }

    thr_info = {"before": before_thr, "after": after_thr}

    # Write report
    write_report(
        m_before,
        m_after,
        hba1c_info,
        p("reports/missingness/missingness_before_after.md"),
        p("reports/missingness/missingness_metrics.json"),
        thr_info=thr_info,
        top_indicators=top_ind,
    )

    print("\n=== Missingness Before/After analysis complete (Phase 2) ===")
    print("Outputs written to phase2_baseline_models/visuals/missingness and reports/missingness")


if __name__ == "__main__":
    main()