import os
import json
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
try:
    from sklearn.model_selection import StratifiedGroupKFold  # type: ignore
    HAS_SGK = True
except Exception:
    HAS_SGK = False

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, brier_score_loss
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin

from joblib import dump

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

try:
    # When executed as a module: python -m SeniorMAC.phase5b.pipeline
    from .utils import prepare_dataset
except ImportError:
    # When executed as a script: python SeniorMAC/phase5b/pipeline.py
    from utils import prepare_dataset


# Optional models
try:
    import lightgbm as lgb  # type: ignore
    HAS_LGB = True
except Exception:
    HAS_LGB = False

try:
    from xgboost import XGBClassifier  # type: ignore
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from catboost import CatBoostClassifier, Pool  # type: ignore
    HAS_CAT = True
except Exception:
    HAS_CAT = False


BASE_DIR = os.path.dirname(__file__)
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
VISUALS_DIR = os.path.join(BASE_DIR, "visuals")
MODELS_DIR = os.path.join(BASE_DIR, "models")
for d in [REPORTS_DIR, VISUALS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

# Calibration method toggle (isotonic reduces extreme probabilities on imbalanced data)
CALIB_METHOD = os.environ.get("PHASE5B_CALIB_METHOD", "isotonic").lower()  # isotonic or sigmoid
# Allow per-model overrides (LR often does better with Platt scaling)
CALIB_METHOD_MAP: Dict[str, str] = {
    "lr": os.environ.get("PHASE5B_CALIB_METHOD_LR", "sigmoid").lower()
}

# Collapse rare categories to reduce spiky logits from sparse OHE
class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    def __init__(self, min_frequency: float = 0.02):
        self.min_frequency = float(min_frequency)
        self.allowed_: Dict[object, set] = {}

    def fit(self, X, y=None):
        import pandas as pd
        if not hasattr(X, "columns"):
            X = pd.DataFrame(X)
        n = len(X)
        threshold = max(1, int(self.min_frequency * n))
        for c in X.columns:
            s = X[c].astype(str)
            vc = s.value_counts()
            allowed = set(vc[vc >= threshold].index.tolist())
            allowed.add("nan")
            self.allowed_[c] = allowed
        return self

    def transform(self, X):
        import pandas as pd
        if not hasattr(X, "columns"):
            X = pd.DataFrame(X)
        X2 = X.copy()
        for c in X2.columns:
            allowed = self.allowed_.get(c, set())
            s = X2[c].astype(str)
            X2[c] = s.where(s.isin(allowed), other="Other")
        return X2


def make_ohe(min_freq: float = 0.02):
    # Safe OneHotEncoder across sklearn versions
    # Prefer dropping very-rare categories to reduce spiky logits
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse=True, min_frequency=min_freq)
    except TypeError:
        # Older sklearn without min_frequency
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def build_preprocessor(cat_cols: List[str], num_cols: List[str]) -> ColumnTransformer:
    num = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])
    rare_freq = float(os.environ.get("PHASE5B_RARE_FREQ", "0.02"))
    cat = Pipeline(steps=[
        ("rare", RareCategoryGrouper(min_frequency=rare_freq)),
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", make_ohe(min_freq=rare_freq)),
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", num, num_cols),
            ("cat", cat, cat_cols),
        ]
    )
    setattr(pre, "_numeric_features", num_cols)
    setattr(pre, "_categorical_features", cat_cols)
    return pre


def ece_score(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    # Expected Calibration Error
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not np.any(mask):
            continue
        avg_conf = y_prob[mask].mean()
        avg_acc = y_true[mask].mean()
        ece += (np.sum(mask) / len(y_true)) * abs(avg_conf - avg_acc)
    return float(ece)


def choose_threshold(y_true: np.ndarray, y_prob: np.ndarray, target_recall: float = 0.85) -> float:
    # Choose smallest threshold achieving target recall; fallback to best F1
    thresholds = np.linspace(0.01, 0.99, 99)
    best_f1, best_t = -1.0, 0.5
    chosen = None
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if chosen is None and rec >= target_recall:
            chosen = t
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(chosen if chosen is not None else best_t)


def bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray, metric_fn, n_boot: int = 200, seed: int = 42) -> Tuple[float, float]:
    rng = np.random.RandomState(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yt = y_true[idx]
        yp = y_prob[idx]
        vals.append(metric_fn(yt, (yp >= choose_threshold(yt, yp)).astype(int)))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def get_models(class_weight: Optional[str] = "balanced") -> Dict[str, object]:
    models: Dict[str, object] = {
        # L1-regularized LR to zero-out spiky OHE coefficients; drop class_weight to reduce saturation
        "lr": LogisticRegression(max_iter=2000, solver="liblinear", class_weight=None, C=0.3, penalty="l1"),
        "rf": RandomForestClassifier(n_estimators=300, random_state=42, class_weight=class_weight),
        "gb": GradientBoostingClassifier(random_state=42),
    }
    if HAS_XGB:
        models["xgb"] = XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.08, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=1.0, random_state=42, n_jobs=-1,
            eval_metric="logloss"
        )
    if HAS_LGB:
        models["lgb"] = lgb.LGBMClassifier(
            n_estimators=600, learning_rate=0.05, num_leaves=31, subsample=0.8,
            colsample_bytree=0.8, random_state=42, n_jobs=-1, class_weight="balanced"
        )
    return models


def run_catboost(X: pd.DataFrame, y: pd.Series, cat_cols: List[str]) -> Tuple[Optional[CatBoostClassifier], Dict[str, float]]:
    if not HAS_CAT:
        return None, {}
    # CatBoost handles cats natively; create pools
    # Prepare categorical columns: CatBoost expects strings/integers, not NaN floats
    X_cb = X.copy()
    for c in cat_cols:
        # ensure string representation; convert NaN to a string token
        X_cb[c] = X_cb[c].astype(str)
        X_cb[c] = X_cb[c].replace({"nan": "nan"}).fillna("nan")
    cat_idx = [X_cb.columns.get_loc(c) for c in cat_cols]
    model = CatBoostClassifier(
        iterations=600,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        random_state=42,
        verbose=False,
        allow_writing_files=False,
        class_weights=[1.0, 1.0]  # rely on fit to balance; can tune if needed
    )
    # Simple CV for CatBoost
    if HAS_SGK:
        # Fallback: no groups available for CatBoost here; use StratifiedKFold
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    probs_all, y_all = [], []
    for tr, te in cv.split(X_cb, y):
        tr_pool = Pool(X_cb.iloc[tr], y.iloc[tr], cat_features=cat_idx)
        te_pool = Pool(X_cb.iloc[te], y.iloc[te], cat_features=cat_idx)
        model.fit(tr_pool)
        prob = model.predict_proba(te_pool)[:, 1]
        probs_all.append(prob)
        y_all.append(y.iloc[te].values)
    y_all = np.concatenate(y_all)
    probs_all = np.concatenate(probs_all)
    thr = choose_threshold(y_all, probs_all)
    metrics = {
        "auc": float(roc_auc_score(y_all, probs_all)),
        "ap": float(average_precision_score(y_all, probs_all)),
    }
    # Fit final model on full data
    final_pool = Pool(X_cb, y, cat_features=cat_idx)
    model.fit(final_pool)
    return model, metrics


def evaluate_cv(pre: ColumnTransformer, models: Dict[str, object], X: pd.DataFrame, y: pd.Series,
                groups: Optional[pd.Series]) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float], Dict[str, Dict[str, np.ndarray]]]:
    # CV setup
    if HAS_SGK and groups is not None:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        splits = list(cv.split(X, y, groups))
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        splits = list(cv.split(X, y))

    metrics_all: Dict[str, Dict[str, float]] = {}
    thresholds: Dict[str, float] = {}
    cv_data: Dict[str, Dict[str, np.ndarray]] = {}

    for name, est in models.items():
        # Calibrate with configured method per model; 5-folds for more stable maps
        method = CALIB_METHOD_MAP.get(name, CALIB_METHOD)
        clf = CalibratedClassifierCV(estimator=est, method=method, cv=5)
        pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
        y_stack, p_stack = [], []
        for tr, te in splits:
            pipe.fit(X.iloc[tr], y.iloc[tr])
            prob = pipe.predict_proba(X.iloc[te])[:, 1]
            y_stack.append(y.iloc[te].values)
            p_stack.append(prob)
        y_all = np.concatenate(y_stack)
        probs = np.clip(np.concatenate(p_stack), 1e-6, 1 - 1e-6)
        thr = choose_threshold(y_all, probs, target_recall=float(os.environ.get("PHASE5B_TARGET_RECALL", "0.85")))
        thresholds[name] = thr
        y_pred = (probs >= thr).astype(int)
        metrics = {
            "accuracy": float(accuracy_score(y_all, y_pred)),
            "precision": float(precision_score(y_all, y_pred, zero_division=0)),
            "recall": float(recall_score(y_all, y_pred, zero_division=0)),
            "f1": float(f1_score(y_all, y_pred, zero_division=0)),
            "auc": float(roc_auc_score(y_all, probs)),
            "ap": float(average_precision_score(y_all, probs)),
            "brier": float(brier_score_loss(y_all, probs)),
            "ece": float(ece_score(y_all, probs)),
        }
        # Bootstrap CI for F1
        lo, hi = bootstrap_ci(y_all, probs, lambda yt, yp: f1_score(yt, yp, zero_division=0), n_boot=int(os.environ.get("PHASE5B_BOOT", "120")))
        metrics["f1_ci_lo"] = float(lo)
        metrics["f1_ci_hi"] = float(hi)
        metrics_all[name] = metrics
        cv_data[name] = {"y": y_all, "prob": probs}
    return metrics_all, thresholds, cv_data


def plot_roc_pr(y_true: np.ndarray, y_prob: np.ndarray, tag: str):
    from sklearn.metrics import roc_curve, precision_recall_curve
    prob = np.clip(y_prob, 1e-6, 1 - 1e-6)
    fpr, tpr, _ = roc_curve(y_true, prob)
    prec, rec, _ = precision_recall_curve(y_true, prob)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(fpr, tpr, label="ROC")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR")
    axes[0].set_title(f"ROC - {tag}")
    axes[1].plot(rec, prec, label="PR")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"PR - {tag}")
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, f"roc_pr_{tag}.svg"), format="svg")
    plt.close()


def compute_shap(pipe: Pipeline, X: pd.DataFrame, tag: str):
    try:
        import shap  # type: ignore
    except Exception:
        return
    # Try to compute SHAP for tree-based final estimator when possible
    try:
        clf = pipe.named_steps["clf"].base_estimator
    except Exception:
        clf = None
    if clf is None:
        return
    # Apply preprocessing to get final feature matrix and names
    pre = pipe.named_steps["pre"]
    Xt = pre.fit_transform(X)
    try:
        feat_names = pre.get_feature_names_out()
    except Exception:
        feat_names = [f"f{i}" for i in range(Xt.shape[1])]
    # Use TreeExplainer if possible
    try:
        explainer = shap.TreeExplainer(clf)
        values = explainer.shap_values(Xt)
        vals = values if isinstance(values, np.ndarray) else values[1]
        imp = np.mean(np.abs(vals), axis=0)
        imp_df = pd.DataFrame({"feature": feat_names, "importance": imp}).sort_values("importance", ascending=False)
        imp_df.to_csv(os.path.join(REPORTS_DIR, f"shap_importance_{tag}.csv"), index=False)
        top = imp_df.head(25)
        plt.figure(figsize=(10, max(3, 0.3 * len(top))))
        sns.barplot(y="feature", x="importance", data=top, color="#4C78A8")
        plt.title(f"SHAP Feature Importance ({tag})")
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALS_DIR, f"shap_importance_{tag}.svg"), format="svg")
        plt.close()
    except Exception:
        return


def main():
    data_path = os.path.abspath(os.path.join(BASE_DIR, "..", "clean_data.csv"))
    df = pd.read_csv(data_path)
    X, y, cat_cols, num_cols, groups = prepare_dataset(df)
    # Drop rows with target NaN
    mask = ~y.isna()
    X, y = X.loc[mask], y.loc[mask]
    if groups is not None:
        groups = groups.loc[mask]

    # Preprocessor
    pre = build_preprocessor(cat_cols, num_cols)

    # Models (excluding CatBoost from OHE pipeline)
    models = get_models(class_weight="balanced")

    # Evaluate via CV (also return out-of-fold predictions for reliable curves)
    metrics_all, thresholds, cv_data = evaluate_cv(pre, models, X, y, groups)
    with open(os.path.join(REPORTS_DIR, "metrics_cv.json"), "w") as f:
        json.dump(metrics_all, f, indent=2)
    with open(os.path.join(REPORTS_DIR, "thresholds.json"), "w") as f:
        json.dump(thresholds, f, indent=2)
    # Persist CV data for diagnostics
    try:
        cv_dump = {k: {"y": v["y"].tolist(), "prob": v["prob"].tolist()} for k, v in cv_data.items()}
        with open(os.path.join(REPORTS_DIR, "cv_predictions.json"), "w") as f:
            json.dump(cv_dump, f, indent=2)
    except Exception:
        pass

    # Fit and save final pipelines
    saved = {}
    for name, est in models.items():
        method = CALIB_METHOD_MAP.get(name, CALIB_METHOD)
        clf = CalibratedClassifierCV(estimator=est, method=method, cv=5)
        pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
        pipe.fit(X, y)
        model_path = os.path.join(MODELS_DIR, f"model_{name}.joblib")
        try:
            dump(pipe, model_path)
            with open(os.path.join(REPORTS_DIR, "save_log.txt"), "a", encoding="utf-8") as f:
                f.write(f"Saved {name} -> {model_path}\n")
            saved[name] = True
        except Exception as e:
            with open(os.path.join(REPORTS_DIR, "save_errors.txt"), "a", encoding="utf-8") as f:
                f.write(f"Error saving {name}: {e}\n")
        # Use CV out-of-fold probabilities for ROC/PR (more realistic, avoids right-angle artifacts)
        try:
            cv_y = cv_data.get(name, {}).get("y")
            cv_p = cv_data.get(name, {}).get("prob")
            if cv_y is not None and cv_p is not None:
                plot_roc_pr(cv_y, cv_p, tag=name)
            else:
                # Fallback to in-sample proba if CV unavailable
                prob = pipe.predict_proba(X)[:, 1]
                plot_roc_pr(y.values, prob, tag=name)
        except Exception:
            try:
                prob = pipe.predict_proba(X)[:, 1]
                plot_roc_pr(y.values, prob, tag=name)
            except Exception:
                pass
        compute_shap(pipe, X, tag=name)

    # CatBoost path (native categorical)
    if HAS_CAT:
        model_cat, cat_metrics = run_catboost(X[cat_cols + num_cols], y, cat_cols)
        if model_cat is not None:
            model_cat.save_model(os.path.join(MODELS_DIR, "model_cat.cbm"))
            with open(os.path.join(REPORTS_DIR, "metrics_cat.json"), "w") as f:
                json.dump(cat_metrics, f, indent=2)
            # Persist feature order to align inference later
            try:
                with open(os.path.join(MODELS_DIR, "model_cat_meta.json"), "w") as f:
                    json.dump({"feature_order": (cat_cols + num_cols)}, f, indent=2)
            except Exception:
                pass

    print("Phase5b completed. Reports, visuals, and models are saved.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        out_path = os.path.join(REPORTS_DIR, "run_error.txt")
        os.makedirs(REPORTS_DIR, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("ERROR: \n")
            f.write(str(e) + "\n\n")
            f.write(traceback.format_exc())
        print(f"Error occurred; details saved to {out_path}")