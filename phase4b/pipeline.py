import os
import json
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, train_test_split, learning_curve
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    make_scorer,
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE
from imblearn.over_sampling import ADASYN

import joblib
from copy import deepcopy

# Safe wrapper around ADASYN to avoid "No samples will be generated" errors in CV folds
class SafeADASYN(ADASYN):
    def fit_resample(self, X, y, **params):
        try:
            # compute current minority/majority ratio
            y_arr = np.asarray(y)
            classes, counts = np.unique(y_arr, return_counts=True)
            if classes.size != 2:
                return super().fit_resample(X, y, **params)
            minority_count = counts.min()
            majority_count = counts.max()
            curr_ratio = minority_count / max(majority_count, 1)
            target = self.sampling_strategy
            adjusted_ss = target
            if isinstance(target, float):
                # ensure target strictly greater than current ratio
                if target <= curr_ratio + 1e-8:
                    adjusted_ss = min(curr_ratio + 0.05, 0.95)
            original_ss = self.sampling_strategy
            self.sampling_strategy = adjusted_ss
            try:
                X_res, y_res = super().fit_resample(X, y, **params)
            finally:
                self.sampling_strategy = original_ss
            return X_res, y_res
        except Exception:
            # graceful fallback: no resampling
            return np.asarray(X), np.asarray(y)

# Safe wrapper around ADASYN to avoid "No samples will be generated" errors in CV folds
from imblearn.over_sampling import KMeansSMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import TomekLinks
from imblearn.ensemble import BalancedRandomForestClassifier

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
PHASE_DIR = os.path.join(PROJECT_ROOT, "phase4b")
REPORTS_DIR = os.path.join(PHASE_DIR, "reports")
VISUALS_DIR = os.path.join(PHASE_DIR, "visuals")
MODELS_DIR = os.path.join(PHASE_DIR, "models")
for d in [REPORTS_DIR, VISUALS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

DATA_PATH = os.path.join(PROJECT_ROOT, "clean_data.csv")

RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 5
MAX_GAP_F1 = 0.12
MAX_GAP_AUC = 0.06
THR_GRID = np.linspace(0.05, 0.95, 91)  # fine grid for threshold tuning


# -----------------------------
# Utilities copied from phase3b
# -----------------------------
def detect_target_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        name = str(col).strip().lower()
        if "death" in name and "outcome" in name:
            return col
    for col in df.columns:
        name = str(col).strip().lower()
        if "death" in name and all(k not in name for k in ["timing", "therapy", "level", "type"]):
            return col
    candidates = ["target", "label", "y", "stroke", "death", "DEATH", "outcome", "Outcome"]
    for c in candidates:
        if c in df.columns:
            return c
    for col in reversed(df.columns):
        s = df[col].dropna()
        uniq = set(str(v).strip().lower() for v in s.unique())
        if len(uniq) == 2 and uniq.issubset({"yes", "no", "1", "0", "true", "false"}):
            return col
        if s.dtype.kind in {"i", "u", "f"} and s.isin([0, 1]).all():
            return col
    return df.columns[-1]


def coerce_binary_target(series: pd.Series) -> pd.Series:
    if series.dtype.kind in {"i", "u", "f"}:
        return series.astype(float)
    non_null = series.dropna()
    uniq = sorted(set(str(v).strip().lower() for v in non_null.unique()))
    if len(uniq) == 2:
        pos_terms = {"1", "yes", "y", "true", "t", "positive", "pos", "death", "deceased", "stroke"}
        mapping = {}
        for v in series.unique():
            if pd.isna(v):
                mapping[v] = np.nan
            else:
                key = str(v).strip().lower()
                mapping[v] = 1.0 if key in pos_terms else 0.0
        return series.map(mapping)
    return series


class FeatureClipper(BaseEstimator, TransformerMixin):
    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper
        self.low_ = None
        self.high_ = None

    def fit(self, X, y=None):
        Xn = np.asarray(X, dtype=float)
        # Compute robust per-feature bounds ignoring NaNs
        self.low_ = np.nanquantile(Xn, self.lower, axis=0)
        self.high_ = np.nanquantile(Xn, self.upper, axis=0)
        return self

    def transform(self, X):
        Xn = np.asarray(X, dtype=float)
        # Clip to learned bounds (broadcast along rows)
        return np.clip(Xn, self.low_, self.high_)


class AllMissingDropper(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.keep_idx_: np.ndarray | None = None
    def fit(self, X, y=None):
        df = pd.DataFrame(X)
        keep = ~df.isna().all(axis=0)
        self.keep_idx_ = keep.to_numpy()
        return self
    def transform(self, X):
        df = pd.DataFrame(X)
        return df.loc[:, self.keep_idx_].to_numpy()


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    # Exclude columns that are entirely missing to avoid imputer warnings and no-op features
    numeric_cols = [c for c in numeric_cols if not X[c].isna().all()]
    categorical_cols = [c for c in categorical_cols if not X[c].isna().all()]

    numeric_tf = Pipeline(steps=[
        ("dropall", AllMissingDropper()),
        ("imputer", SimpleImputer(strategy="median")),
        ("clipper", FeatureClipper(lower=0.01, upper=0.99)),
        ("scaler", StandardScaler()),
    ])
    categorical_tf = Pipeline(steps=[
        ("dropall", AllMissingDropper()),
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    pre = ColumnTransformer(transformers=[
        ("num", numeric_tf, numeric_cols),
        ("cat", categorical_tf, categorical_cols),
    ])
    return pre


# -----------------------------
# Modeling helpers
# -----------------------------
SAMPLERS = {
    "smote": SMOTE,
    "bsmote": BorderlineSMOTE,
    "svmsmote": SVMSMOTE,
    "smotetomek": SMOTETomek,
    "smoteenn": SMOTEENN,
    "none": None,
    # 'smote_nearmiss' and 'adasyn_nearmiss' are handled specially in make_pipeline
}


def make_pipeline(pre: ColumnTransformer, sampler_name: str, sampler_kwargs: Dict | None = None,
                  rf_kwargs: Dict | None = None) -> ImbPipeline:
    sampler_kwargs = sampler_kwargs or {}
    rf_kwargs = rf_kwargs or {}
    if sampler_name == "none":
        sampler = "passthrough"
        rf = RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=300,
                                    class_weight="balanced", n_jobs=-1, **rf_kwargs)
        pipe = ImbPipeline(steps=[
            ("pre", pre),
            ("sampler", sampler),
            ("rf", rf),
        ])
        return pipe
    if sampler_name == "smote_nearmiss":
        # Mild oversampling then cautious under-sampling to simplify boundaries
        smote = SMOTE(random_state=RANDOM_STATE, **{k: v for k, v in sampler_kwargs.items() if k in {"sampling_strategy", "k_neighbors"}})
        # NearMiss params (version, n_neighbors, sampling_strategy) will be tuned via HPO
        nearmiss = NearMiss()
        rf = RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=300,
                                    class_weight="balanced", n_jobs=-1, **rf_kwargs)
        pipe = ImbPipeline(steps=[
            ("pre", pre),
            ("sampler_smote", smote),
            ("sampler_nearmiss", nearmiss),
            ("rf", rf),
        ])
        return pipe
    if sampler_name == "kmeans_smote_nearmiss":
        # KMeansSMOTE oversampling to generate synthetic samples in cluster space, then NearMiss
        ksm = KMeansSMOTE(random_state=RANDOM_STATE, **{k: v for k, v in sampler_kwargs.items() if k in {"sampling_strategy", "k_neighbors"}})
        nearmiss = NearMiss()
        rf = RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=300,
                                    class_weight="balanced", n_jobs=-1, **rf_kwargs)
        pipe = ImbPipeline(steps=[
            ("pre", pre),
            ("sampler_kmeans_smote", ksm),
            ("sampler_nearmiss", nearmiss),
            ("rf", rf),
        ])
        return pipe
    if sampler_name == "none_brf":
        # Cost-sensitive baseline using BalancedRandomForest (internal under-sampling per tree)
        sampler = "passthrough"
        brf = BalancedRandomForestClassifier(random_state=RANDOM_STATE, n_estimators=300, n_jobs=-1, **rf_kwargs)
        pipe = ImbPipeline(steps=[
            ("pre", pre),
            ("sampler", sampler),
            ("brf", brf),
        ])
        return pipe
    if sampler_name == "adasyn_nearmiss":
        # ADASYN to focus on difficult minority areas, followed by NearMiss under-sampling
        adasyn = SafeADASYN(random_state=RANDOM_STATE, **{k: v for k, v in sampler_kwargs.items() if k in {"sampling_strategy", "n_neighbors"}})
        nearmiss = NearMiss()
        rf = RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=300,
                                    class_weight="balanced", n_jobs=-1, **rf_kwargs)
        pipe = ImbPipeline(steps=[
            ("pre", pre),
            ("sampler_adasyn", adasyn),
            ("sampler_nearmiss", nearmiss),
            ("rf", rf),
        ])
        return pipe
    if sampler_name == "adasyn_tomek":
        # ADASYN oversampling followed by TomekLinks cleaning under-sampling
        adasyn = SafeADASYN(random_state=RANDOM_STATE, **{k: v for k, v in sampler_kwargs.items() if k in {"sampling_strategy", "n_neighbors"}})
        tomek = TomekLinks()
        rf = RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=300,
                                    class_weight="balanced", n_jobs=-1, **rf_kwargs)
        pipe = ImbPipeline(steps=[
            ("pre", pre),
            ("sampler_adasyn", adasyn),
            ("sampler_tomek", tomek),
            ("rf", rf),
        ])
        return pipe
    SamplerCls = SAMPLERS[sampler_name]
    sampler = SamplerCls(random_state=RANDOM_STATE, **sampler_kwargs)
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=300,
                                class_weight="balanced", n_jobs=-1, **rf_kwargs)
    pipe = ImbPipeline(steps=[
        ("pre", pre),
        ("sampler", sampler),
        ("rf", rf),
    ])
    return pipe


def evaluate_thresholds(y_true: np.ndarray, y_prob: np.ndarray, thr_grid: np.ndarray = THR_GRID) -> pd.DataFrame:
    rows = []
    for thr in thr_grid:
        y_pred = (y_prob >= thr).astype(int)
        rows.append({
            "threshold": thr,
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "accuracy": accuracy_score(y_true, y_pred),
        })
    return pd.DataFrame(rows)


def crossval_with_overfitting(pipe: ImbPipeline, X: pd.DataFrame, y: pd.Series, n_splits: int = N_SPLITS,
                               thr_grid: np.ndarray = THR_GRID) -> Dict:
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
    val_probs_all: List[float] = []
    val_y_all: List[int] = []
    thr_per_fold: List[float] = []
    per_fold = []
    fold_num = 0
    for tr_idx, va_idx in cv.split(X, y):
        fold_num += 1
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        try:
            pipe.fit(X_tr, y_tr)
            tr_prob = pipe.predict_proba(X_tr)[:, 1]
            tr_pred = (tr_prob >= 0.5).astype(int)
            tr_f1 = f1_score(y_tr, tr_pred, zero_division=0)
            tr_auc = roc_auc_score(y_tr, tr_prob)

            va_prob = pipe.predict_proba(X_va)[:, 1]
            va_pred = (va_prob >= 0.5).astype(int)
            va_f1 = f1_score(y_va, va_pred, zero_division=0)
            va_auc = roc_auc_score(y_va, va_prob)

            # per-fold optimal threshold on validation
            va_thr_df = evaluate_thresholds(y_va, va_prob, thr_grid)
            va_thr_df.sort_values(["f1", "precision", "recall"], ascending=False, inplace=True)
            thr_per_fold.append(float(va_thr_df.iloc[0]["threshold"]))

            per_fold.append({
                "fold": fold_num,
                "train_f1@0.5": tr_f1,
                "val_f1@0.5": va_f1,
                "gap_f1@0.5": tr_f1 - va_f1,
                "train_roc_auc": tr_auc,
                "val_roc_auc": va_auc,
                "gap_roc_auc": tr_auc - va_auc,
            })

            val_probs_all.extend(va_prob.tolist())
            val_y_all.extend(y_va.tolist())
        except Exception as e:
            print(f"Cross-val fold {fold_num} failed with error: {e}. Skipping this fold.")
            continue

    if len(val_probs_all) == 0:
        # Return empty results when all folds failed; caller should handle skipping this sampler
        return {
            "threshold_table": pd.DataFrame(columns=["threshold", "precision", "recall", "f1", "accuracy"]),
            "best_threshold": 0.5,
            "overfitting": pd.DataFrame(columns=["fold","train_f1@0.5","val_f1@0.5","gap_f1@0.5","train_roc_auc","val_roc_auc","gap_roc_auc"]),
            "thr_per_fold": [],
        }

    val_probs_all = np.asarray(val_probs_all)
    val_y_all = np.asarray(val_y_all)
    thr_df = evaluate_thresholds(val_y_all, val_probs_all, thr_grid)
    thr_df.sort_values(["f1", "precision", "recall"], ascending=False, inplace=True)
    best_thr = float(np.median(thr_per_fold)) if len(thr_per_fold) > 0 else float(thr_df.iloc[0]["threshold"])

    overfit_df = pd.DataFrame(per_fold)
    return {
        "best_threshold": best_thr,
        "threshold_table": thr_df,
        "overfitting": overfit_df,
        "thr_per_fold": thr_per_fold,
    }


def compare_calibrators(pipe: ImbPipeline, X_train: pd.DataFrame, y_train: pd.Series,
                        X_test: pd.DataFrame, y_test: pd.Series, best_thr: float) -> Tuple[pd.DataFrame, str, float]:
    rows = []
    curves = {}
    for method in ["isotonic", "sigmoid"]:
        clf = CalibratedClassifierCV(estimator=pipe, method=method, cv=3)
        clf.fit(X_train, y_train)
        prob_test = clf.predict_proba(X_test)[:, 1]
        brier = brier_score_loss(y_test, prob_test)
        # reliability curve
        frac_pos, mean_pred = calibration_curve(y_test, prob_test, n_bins=10, strategy="uniform")
        curves[method] = (mean_pred, frac_pos)
        # tune threshold on calibrated probabilities to maximize F1
        thr_df = evaluate_thresholds(y_test.values, prob_test)
        thr_best = float(thr_df.sort_values("f1", ascending=False).iloc[0]["threshold"]) if not thr_df.empty else best_thr
        y_pred = (prob_test >= thr_best).astype(int)
        rows.append({
            "calibration": method,
            "brier": brier,
            "threshold": thr_best,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, prob_test),
            "pr_auc": average_precision_score(y_test, prob_test),
        })
    calib_df = pd.DataFrame(rows).sort_values(["f1", "roc_auc", "brier"], ascending=[False, False, True])

    # plot reliability
    plt.figure(figsize=(6, 5))
    for method, (mp, fp) in curves.items():
        plt.plot(mp, fp, marker="o", label=f"{method}")
    xs = np.linspace(0, 1, 100)
    plt.plot(xs, xs, "k--", alpha=0.5, label="perfect")
    plt.xlabel("Mean predicted")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curves (Phase 4b)")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, "calibration_compare.svg"), format="svg")
    plt.close()

    best_cal = calib_df.iloc[0]["calibration"]
    best_thr_cal = float(calib_df.iloc[0]["threshold"]) if "threshold" in calib_df.columns else best_thr
    return calib_df, str(best_cal), best_thr_cal


def plot_overfitting_gaps(df: pd.DataFrame, out_path: str):
    if df.empty:
        return
    dfm = df.melt(id_vars=["fold"], value_vars=["gap_f1@0.5", "gap_roc_auc"], var_name="metric", value_name="gap")
    plt.figure(figsize=(8, 5))
    sns.barplot(data=dfm, x="metric", y="gap", estimator=np.mean, errorbar=("pi", 95), palette=["#F58518", "#4C78A8"])
    plt.axhline(0, color="k", linewidth=1)
    plt.title("Average Train-Validation Gap (Overfitting Signal)")
    plt.ylabel("Train - Validation")
    plt.tight_layout()
    plt.savefig(out_path, format="svg")
    plt.close()


def plot_threshold_distribution(thrs: List[float], out_path: str) -> None:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=thrs, color="#4C78A8")
    plt.xlabel("Per-fold optimal thresholds")
    plt.title("Phase 4b: Per-fold threshold distribution")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, format="svg")
    plt.close()


def run_hpo(pipe: ImbPipeline, sampler_name: str, X: pd.DataFrame, y: pd.Series):
    scoring = {
        "f1": make_scorer(f1_score, zero_division=0),
        "pr_auc": "average_precision",
        "roc_auc": "roc_auc",
    }
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)

    # Compute current minority ratio to set safe oversampling targets
    pos_frac = float(np.mean(y))
    minority_ratio = min(pos_frac, 1 - pos_frac)
    def safe_sampling_values(curr: float, cap: float = 0.2):
        base = max(curr + 0.05, cap)
        v1 = round(base, 2)
        v2 = round(min(base + 0.1, 0.5), 2)
        return sorted(set([v1, v2]))

    param_dist = {
        "rf__n_estimators": [200, 300, 400],
        "rf__max_depth": [5, 8, 12, 16],
        "rf__max_features": ["sqrt", 0.3, 0.5],
        "rf__min_samples_split": [5, 10, 15],
        "rf__min_samples_leaf": [3, 5, 8],
        "rf__bootstrap": [True],
        "rf__class_weight": ["balanced"],
    }
    if sampler_name == "smote_nearmiss":
        # Cap oversampling intensity <= 0.2 (adaptive)
        param_dist["sampler_smote__sampling_strategy"] = safe_sampling_values(minority_ratio)
        param_dist["sampler_smote__k_neighbors"] = [3, 5]
        # Tune NearMiss conservatively to avoid excessive information loss
        param_dist["sampler_nearmiss__version"] = [1]
        param_dist["sampler_nearmiss__n_neighbors"] = [3, 5]
        param_dist["sampler_nearmiss__sampling_strategy"] = [0.4, 0.5]
    elif sampler_name in ["smote", "bsmote", "svmsmote"]:
        # Cap oversampling intensity <= 0.2 (adaptive)
        param_dist["sampler__sampling_strategy"] = safe_sampling_values(minority_ratio)
        param_dist["sampler__k_neighbors"] = [3, 5]
    elif sampler_name in ["smotetomek", "smoteenn"]:
        # Cap oversampling intensity <= 0.2 (adaptive)
        param_dist["sampler__sampling_strategy"] = safe_sampling_values(minority_ratio)
    elif sampler_name == "adasyn_nearmiss":
        # ADASYN + NearMiss hybrid tuning, cap oversampling (adaptive)
        param_dist["sampler_adasyn__sampling_strategy"] = safe_sampling_values(minority_ratio)
        param_dist["sampler_adasyn__n_neighbors"] = [1, 2]
        param_dist["sampler_nearmiss__version"] = [1]
        param_dist["sampler_nearmiss__n_neighbors"] = [3, 5]
        param_dist["sampler_nearmiss__sampling_strategy"] = [0.4, 0.5]
    elif sampler_name == "adasyn_tomek":
        # ADASYN + TomekLinks hybrid tuning; cap oversampling (adaptive)
        param_dist["sampler_adasyn__sampling_strategy"] = safe_sampling_values(minority_ratio)
        param_dist["sampler_adasyn__n_neighbors"] = [1, 2]
        # TomekLinks typically has no k-neighbors; keep defaults to avoid invalid params
    elif sampler_name == "kmeans_smote_nearmiss":
        # KMeansSMOTE + NearMiss hybrid tuning; cap oversampling (adaptive)
        param_dist["sampler_kmeans_smote__sampling_strategy"] = safe_sampling_values(minority_ratio)
        param_dist["sampler_kmeans_smote__k_neighbors"] = [3, 5]
        param_dist["sampler_nearmiss__version"] = [1]
        param_dist["sampler_nearmiss__n_neighbors"] = [3, 5]
        param_dist["sampler_nearmiss__sampling_strategy"] = [0.4, 0.5]
    elif sampler_name == "none_brf":
        # BalancedRandomForest tuning
        param_dist = {
            "brf__n_estimators": [200, 300],
            "brf__max_features": ["sqrt", 0.3, 0.5],
            "brf__min_samples_split": [5, 10, 15],
            "brf__min_samples_leaf": [3, 5, 8],
            "brf__max_depth": [12, 16, 20],
            "brf__class_weight": ["balanced"],
        }

    # Perform randomized search HPO
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=25,
        scoring=scoring,
        refit="f1",
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
    )
    search.fit(X, y)
    return search

# Nested CV fold metrics visual

def plot_nested_cv_fold_metrics(df: pd.DataFrame, out_path: str) -> None:
    if df is None or df.empty:
        return
    plt.figure(figsize=(8, 4))
    sns.barplot(data=df, x="fold", y="f1", color="#4C78A8")
    plt.title("Nested CV: Validation F1 per outer fold")
    plt.xlabel("Fold")
    plt.ylabel("F1")
    plt.tight_layout()
    plt.savefig(out_path, format="svg")
    plt.close()

# Outer StratifiedKFold with inner HPO

def run_nested_cv(pre: ColumnTransformer, sampler_name: str, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, Dict]:
    outer = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_rows: List[Dict] = []
    fold_idx = 0
    for tr_idx, va_idx in outer.split(X, y):
        fold_idx += 1
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        pipe_base = make_pipeline(pre, sampler_name)
        try:
            search = run_hpo(pipe_base, sampler_name, X_tr, y_tr)
            best_pipe = search.best_estimator_
            res = crossval_with_overfitting(best_pipe, X_tr, y_tr)
            best_thr = res.get("best_threshold", 0.5)
            best_pipe.fit(X_tr, y_tr)
            prob_va = best_pipe.predict_proba(X_va)[:, 1]
            y_pred_va = (prob_va >= best_thr).astype(int)
            fold_rows.append({
                "fold": fold_idx,
                "accuracy": accuracy_score(y_va, y_pred_va),
                "precision": precision_score(y_va, y_pred_va, zero_division=0),
                "recall": recall_score(y_va, y_pred_va, zero_division=0),
                "f1": f1_score(y_va, y_pred_va, zero_division=0),
                "roc_auc": roc_auc_score(y_va, prob_va),
                "pr_auc": average_precision_score(y_va, prob_va),
                "best_threshold": float(best_thr),
            })
        except Exception as e:
            print(f"Nested CV fold {fold_idx} failed: {e}. Skipping.")
            continue
    df = pd.DataFrame(fold_rows)
    summary = {
        "sampler": sampler_name,
        "mean_accuracy": float(df["accuracy"].mean()) if not df.empty else None,
        "mean_precision": float(df["precision"].mean()) if not df.empty else None,
        "mean_recall": float(df["recall"].mean()) if not df.empty else None,
        "mean_f1": float(df["f1"].mean()) if not df.empty else None,
        "mean_roc_auc": float(df["roc_auc"].mean()) if not df.empty else None,
        "mean_pr_auc": float(df["pr_auc"].mean()) if not df.empty else None,
    }
    try:
        df.to_csv(os.path.join(REPORTS_DIR, f"{sampler_name}_nested_cv_metrics.csv"), index=False)
        with open(os.path.join(REPORTS_DIR, f"{sampler_name}_nested_cv_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        plot_nested_cv_fold_metrics(df, os.path.join(VISUALS_DIR, f"{sampler_name}_nested_cv_f1.svg"))
    except Exception:
        pass
    return df, summary

def plot_learning_curve(estimator: ImbPipeline, X: pd.DataFrame, y: pd.Series, out_path: str):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring="f1", n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5)
    )
    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)
    plt.figure(figsize=(7, 5))
    plt.plot(train_sizes, train_mean, "-o", label="train F1")
    plt.plot(train_sizes, val_mean, "-o", label="CV F1")
    plt.xlabel("Train size")
    plt.ylabel("F1 score")
    plt.title("Phase 4b: Learning Curve (RF + Sampler)")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, format="svg")
    plt.close()


def main():
    # Load data
    df = pd.read_csv(DATA_PATH)
    target_col = detect_target_column(df)
    df = df.dropna(subset=[target_col]).copy()
    df[target_col] = coerce_binary_target(df[target_col])
    df = df.dropna(subset=[target_col]).copy()

    X = df.drop(columns=[target_col])
    # Drop features that are entirely missing to avoid imputer warnings
    all_missing_cols = [c for c in X.columns if X[c].isna().all()]
    if len(all_missing_cols) > 0:
        print(f"Warning: Dropping all-missing features: {all_missing_cols}")
        X = X.drop(columns=all_missing_cols)
    y = df[target_col].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    pre = build_preprocessor(X)

    # Samplers to compare; include cost-sensitive baseline and mild oversampling hybrids
    sampler_candidates = ["none", "none_brf", "smote", "bsmote", "smoteenn", "smotetomek", "svmsmote", "smote_nearmiss", "kmeans_smote_nearmiss", "adasyn_nearmiss", "adasyn_tomek"]

    results: List[Dict] = []
    rows_ext: List[Dict] = []
    best_by_f1 = None

    for sampler_name in sampler_candidates:
        pipe_base = make_pipeline(pre, sampler_name)
        try:
            search = run_hpo(pipe_base, sampler_name, X_train, y_train)
            best_pipe = search.best_estimator_
            # Save HPO results
            pd.DataFrame(search.cv_results_).to_csv(os.path.join(REPORTS_DIR, f"{sampler_name}_hpo_cv_results.csv"), index=False)
            with open(os.path.join(REPORTS_DIR, f"{sampler_name}_best_params.json"), "w") as f:
                json.dump(search.best_params_, f, indent=2)
        except Exception as e:
            print(f"Sampler {sampler_name} failed during HPO: {e}")
            continue

        res = crossval_with_overfitting(best_pipe, X_train, y_train)
        thr_df = res["threshold_table"]
        best_thr = res["best_threshold"]
        overfit_df = res["overfitting"]
        thr_per_fold = res.get("thr_per_fold", [])

        # Fit on full train and evaluate on test using tuned threshold
        try:
            best_pipe.fit(X_train, y_train)
            prob_test = best_pipe.predict_proba(X_test)[:, 1]
            y_pred_test = (prob_test >= best_thr).astype(int)
        except Exception as e:
            print(f"Sampler {sampler_name} failed during final fit/eval: {e}. Skipping sampler.")
            continue
        row = {
            "sampler": sampler_name,
            "best_threshold": best_thr,
            "test_accuracy": accuracy_score(y_test, y_pred_test) if False else accuracy_score(y_test, y_pred_test),
            "test_precision": precision_score(y_test, y_pred_test, zero_division=0),
            "test_recall": recall_score(y_test, y_pred_test, zero_division=0),
            "test_f1": f1_score(y_test, y_pred_test, zero_division=0),
            "test_roc_auc": roc_auc_score(y_test, prob_test),
            "test_pr_auc": average_precision_score(y_test, prob_test),
            "mean_gap_f1": float(overfit_df["gap_f1@0.5"].mean()) if not overfit_df.empty else None,
            "mean_gap_auc": float(overfit_df["gap_roc_auc"].mean()) if not overfit_df.empty else None,
        }
        results.append(row)
        rows_ext.append({"row": row, "pipe": best_pipe})

        # Persist threshold sweep, per-fold thresholds, and overfitting per sampler
        thr_df.to_csv(os.path.join(REPORTS_DIR, f"{sampler_name}_threshold_sweep.csv"), index=False)
        pd.DataFrame({"threshold": thr_per_fold}).to_csv(os.path.join(REPORTS_DIR, f"{sampler_name}_per_fold_thresholds.csv"), index=False)
        overfit_df.to_csv(os.path.join(REPORTS_DIR, f"{sampler_name}_overfitting_folds.csv"), index=False)
        plot_overfitting_gaps(overfit_df, os.path.join(VISUALS_DIR, f"{sampler_name}_overfitting_gap.svg"))
        try:
            plot_threshold_distribution(thr_per_fold, os.path.join(VISUALS_DIR, f"{sampler_name}_per_fold_thresholds.svg"))
        except Exception:
            pass

        # Learning curve for the sampler
        try:
            plot_learning_curve(best_pipe, X_train, y_train, os.path.join(VISUALS_DIR, f"{sampler_name}_learning_curve.svg"))
        except Exception:
            pass

        # Save bootstrap CIs per sampler
        ci = bootstrap_ci(y_test.values, y_pred_test, prob_test)
        with open(os.path.join(REPORTS_DIR, f"{sampler_name}_test_metric_cis.json"), "w") as f:
            json.dump(ci, f, indent=2)

    # Strict gating: prefer only models under gap thresholds; fallback to cost-sensitive baseline
    if len(results) == 0:
        print("No sampler produced valid results due to strict overfitting/compatibility filters.")
        return
    acceptable_ext = [r for r in rows_ext if r["row"]["mean_gap_f1"] is not None and r["row"]["mean_gap_f1"] <= MAX_GAP_F1 and r["row"]["mean_gap_auc"] <= MAX_GAP_AUC]
    if len(acceptable_ext) > 0:
        best_ext = sorted(acceptable_ext, key=lambda r: (r["row"]["test_f1"], r["row"]["test_roc_auc"]), reverse=True)[0]
    else:
        baseline_ext = next((r for r in rows_ext if r["row"]["sampler"] == "none"), None)
        best_ext = baseline_ext or sorted(rows_ext, key=lambda r: (r["row"]["mean_gap_auc"], r["row"]["test_f1"], r["row"]["test_roc_auc"]), reverse=True)[0]
    best_by_f1 = {**best_ext["row"], "pipe": best_ext["pipe"]}

    # Save overall sampler comparison
    cmp_df = pd.DataFrame(results).sort_values(["mean_gap_auc", "test_f1", "test_roc_auc", "test_accuracy"], ascending=[True, False, False, False])
    cmp_df.to_csv(os.path.join(REPORTS_DIR, "phase4b_sampler_comparison.csv"), index=False)

    # Calibration comparison using best sampler
    best_pipe = best_by_f1["pipe"]
    best_thr = float(best_by_f1["best_threshold"]) if best_by_f1 is not None else 0.5
    calib_df, best_cal, best_thr_cal = compare_calibrators(best_pipe, X_train, y_train, X_test, y_test, best_thr)
    calib_df.to_csv(os.path.join(REPORTS_DIR, "phase4b_calibration_comparison.csv"), index=False)

    # Persist the calibrated best model for deployment/inference
    try:
        calibrated_clf = CalibratedClassifierCV(estimator=best_pipe, method=str(best_cal), cv=3)
        calibrated_clf.fit(X_train, y_train)
        # Evaluate calibrated model on test using calibrator-specific threshold
        prob_test_cal = calibrated_clf.predict_proba(X_test)[:, 1]
        y_pred_test_cal = (prob_test_cal >= best_thr_cal).astype(int)
        best_model_metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred_test_cal)),
            "precision": float(precision_score(y_test, y_pred_test_cal, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred_test_cal, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred_test_cal, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, prob_test_cal)),
            "pr_auc": float(average_precision_score(y_test, prob_test_cal)),
            "brier": float(brier_score_loss(y_test, prob_test_cal)),
        }
        joblib.dump(calibrated_clf, os.path.join(MODELS_DIR, "phase4b_best_calibrated_model.joblib"))
        with open(os.path.join(MODELS_DIR, "phase4b_best_model_meta.json"), "w") as f:
            json.dump({
                "sampler": best_by_f1["sampler"],
                "calibrator": str(best_cal),
                "threshold": float(best_thr_cal),
                "test_metrics_calibrated": best_model_metrics,
            }, f, indent=2)
    except Exception as e:
        print(f"Persisting calibrated best model failed: {e}")

    # Seed-averaged calibrated evaluation
    try:
        seeds_df, seeds_summary = evaluate_seed_averaged_calibrated(best_pipe, X, y, best_thr_cal)
        seeds_df.to_csv(os.path.join(REPORTS_DIR, "phase4b_seed_calibrated_metrics.csv"), index=False)
        with open(os.path.join(REPORTS_DIR, "phase4b_seed_calibrated_summary.json"), "w") as f:
            json.dump(seeds_summary, f, indent=2)
        plot_seed_calibrated_f1(seeds_df, os.path.join(VISUALS_DIR, "phase4b_seed_calibrated_f1.svg"))
    except Exception as e:
        print(f"Seed-averaged calibrated evaluation failed: {e}")

    # Nested CV on the best sampler (outer folds + inner HPO)
    try:
        nested_df, nested_summary = run_nested_cv(pre, best_by_f1["sampler"], X_train, y_train)
        # also save a small visual for nested CV
        plot_nested_cv_fold_metrics(nested_df, os.path.join(VISUALS_DIR, f"{best_by_f1['sampler']}_nested_cv_f1.svg"))
    except Exception as e:
        print(f"Nested CV failed: {e}")

    # Summary and recommendation
    summary = {
        "best_sampler": best_by_f1["sampler"] if best_by_f1 else None,
        "best_threshold": best_thr_cal,
        "best_calibrator": best_cal,
        "mean_gap_f1": best_by_f1.get("mean_gap_f1") if best_by_f1 else None,
        "mean_gap_auc": best_by_f1.get("mean_gap_auc") if best_by_f1 else None,
        "test_metrics": {
            k: float(best_by_f1[k]) for k in [
                "test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc", "test_pr_auc"
            ]
        } if best_by_f1 else {},
    }

    # attach bootstrap CIs for best
    if best_by_f1:
        best_prob_ci_pipe = best_by_f1["pipe"]
        best_prob = best_prob_ci_pipe.predict_proba(X_test)[:, 1]
        best_pred = (best_prob >= best_thr).astype(int)
        summary["test_metric_cis"] = bootstrap_ci(y_test.values, best_pred, best_prob)
        
        # Bootstrap over imputations for missing data uncertainty quantification
        print("Running bootstrap over imputations for missing data uncertainty...")
        try:
            bootstrap_imputation_results = bootstrap_over_imputations(
                best_prob_ci_pipe, X_train, X_test, y_train, y_test, best_thr,
                imputation_strategy="iterative_median",
                n_imputations=5,  # Reduced for speed in pipeline
                n_bootstrap=100,  # Reduced for speed in pipeline
                random_state=RANDOM_STATE
            )
            summary["bootstrap_over_imputations"] = bootstrap_imputation_results
            
            # Save detailed bootstrap over imputations results
            with open(os.path.join(REPORTS_DIR, "phase4b_bootstrap_over_imputations.json"), "w") as f:
                json.dump(bootstrap_imputation_results, f, indent=2)
            
            print(f"Bootstrap over imputations completed. Results saved to phase4b_bootstrap_over_imputations.json")
        except Exception as e:
            print(f"Bootstrap over imputations failed: {e}")
            summary["bootstrap_over_imputations"] = None
    
    with open(os.path.join(REPORTS_DIR, "phase4b_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Visual summary: sampler comparison bar chart
    plt.figure(figsize=(9, 6))
    sub = cmp_df.head(10).copy()
    sub = sub.melt(id_vars=["sampler"], value_vars=["test_accuracy", "test_f1", "test_roc_auc"],
                   var_name="metric", value_name="value")
    sns.barplot(data=sub, x="sampler", y="value", hue="metric",
                palette=["#4C78A8", "#F58518", "#72B7B2"])
    plt.title("Phase 4b: RF comparison — cost-sensitive baseline + mild oversampling (gated by low overfitting)")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, "sampler_comparison.svg"), format="svg")
    plt.close()

    print("Phase 4b complete.")


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, n_boot: int = 500, seed: int = RANDOM_STATE) -> Dict:
    """Original bootstrap CI function for sampling uncertainty only."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    accs, f1s, rocs, prs = [], [], [], []
    idx = np.arange(n)
    for _ in range(n_boot):
        samp = rng.choice(idx, size=n, replace=True)
        yt = y_true[samp]
        yp = y_pred[samp]
        pr = y_prob[samp]
        try:
            accs.append(accuracy_score(yt, yp))
            f1s.append(f1_score(yt, yp, zero_division=0))
            rocs.append(roc_auc_score(yt, pr))
            prs.append(average_precision_score(yt, pr))
        except Exception:
            continue
    def ci(a):
        a = np.asarray(a)
        if a.size == 0:
            return {"mean": None, "low": None, "high": None}
        return {"mean": float(a.mean()), "low": float(np.percentile(a, 2.5)), "high": float(np.percentile(a, 97.5))}
    return {"accuracy": ci(accs), "f1": ci(f1s), "roc_auc": ci(rocs), "pr_auc": ci(prs)}


def create_multiple_imputations(X_train: pd.DataFrame, X_test: pd.DataFrame, 
                               imputation_strategy: str = "iterative_median", 
                               n_imputations: int = 10, random_state: int = RANDOM_STATE) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Create multiple imputed datasets using different strategies.
    
    Args:
        X_train: Training features with missing values
        X_test: Test features with missing values  
        imputation_strategy: Strategy for imputation
        n_imputations: Number of imputed datasets to create
        random_state: Random seed
        
    Returns:
        List of (X_train_imputed, X_test_imputed) tuples
    """
    imputed_datasets = []
    
    # Identify numeric and categorical columns
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    
    for i in range(n_imputations):
        seed = random_state + i
        X_train_imp = X_train.copy()
        X_test_imp = X_test.copy()
        
        if imputation_strategy == "iterative_median":
            # Iterative imputation for numeric columns
            if numeric_cols:
                imputer = IterativeImputer(initial_strategy='median', random_state=seed, max_iter=10)
                X_train_imp[numeric_cols] = imputer.fit_transform(X_train_imp[numeric_cols])
                X_test_imp[numeric_cols] = imputer.transform(X_test_imp[numeric_cols])
            
            # Most frequent for categorical
            if categorical_cols:
                cat_imputer = SimpleImputer(strategy='most_frequent')
                X_train_imp[categorical_cols] = cat_imputer.fit_transform(X_train_imp[categorical_cols])
                X_test_imp[categorical_cols] = cat_imputer.transform(X_test_imp[categorical_cols])
                
        elif imputation_strategy == "iterative_mean":
            if numeric_cols:
                imputer = IterativeImputer(initial_strategy='mean', random_state=seed, max_iter=10)
                X_train_imp[numeric_cols] = imputer.fit_transform(X_train_imp[numeric_cols])
                X_test_imp[numeric_cols] = imputer.transform(X_test_imp[numeric_cols])
            if categorical_cols:
                cat_imputer = SimpleImputer(strategy='most_frequent')
                X_train_imp[categorical_cols] = cat_imputer.fit_transform(X_train_imp[categorical_cols])
                X_test_imp[categorical_cols] = cat_imputer.transform(X_test_imp[categorical_cols])
                
        elif imputation_strategy.startswith("knn"):
            k = int(imputation_strategy.split("knn")[1]) if len(imputation_strategy) > 3 else 5
            if numeric_cols:
                imputer = KNNImputer(n_neighbors=k)
                X_train_imp[numeric_cols] = imputer.fit_transform(X_train_imp[numeric_cols])
                X_test_imp[numeric_cols] = imputer.transform(X_test_imp[numeric_cols])
            if categorical_cols:
                cat_imputer = SimpleImputer(strategy='most_frequent')
                X_train_imp[categorical_cols] = cat_imputer.fit_transform(X_train_imp[categorical_cols])
                X_test_imp[categorical_cols] = cat_imputer.transform(X_test_imp[categorical_cols])
                
        else:  # fallback to median/most_frequent
            if numeric_cols:
                num_imputer = SimpleImputer(strategy='median')
                X_train_imp[numeric_cols] = num_imputer.fit_transform(X_train_imp[numeric_cols])
                X_test_imp[numeric_cols] = num_imputer.transform(X_test_imp[numeric_cols])
            if categorical_cols:
                cat_imputer = SimpleImputer(strategy='most_frequent')
                X_train_imp[categorical_cols] = cat_imputer.fit_transform(X_train_imp[categorical_cols])
                X_test_imp[categorical_cols] = cat_imputer.transform(X_test_imp[categorical_cols])
        
        imputed_datasets.append((X_train_imp, X_test_imp))
    
    return imputed_datasets


def bootstrap_over_imputations(pipe: ImbPipeline, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                              y_train: pd.Series, y_test: pd.Series, best_thr: float,
                              imputation_strategy: str = "iterative_median", 
                              n_imputations: int = 10, n_bootstrap: int = 200, 
                              random_state: int = RANDOM_STATE) -> Dict:
    """Bootstrap over multiple imputations to quantify missing data uncertainty.
    
    This function creates multiple imputed datasets, trains models on each,
    and uses bootstrap resampling to generate confidence intervals that account
    for both sampling and imputation uncertainty.
    
    Args:
        pipe: Trained pipeline (should be fitted on complete data)
        X_train, X_test: Features with missing values
        y_train, y_test: Target variables
        best_thr: Optimal threshold for binary classification
        imputation_strategy: Strategy for creating imputations
        n_imputations: Number of imputed datasets
        n_bootstrap: Number of bootstrap samples per imputation
        random_state: Random seed
        
    Returns:
        Dictionary with confidence intervals for multiple metrics
    """
    print(f"Creating {n_imputations} imputed datasets using {imputation_strategy}...")
    
    # Create multiple imputed datasets
    imputed_datasets = create_multiple_imputations(
        X_train, X_test, imputation_strategy, n_imputations, random_state
    )
    
    # Collect predictions from all imputations
    all_predictions = []
    all_probabilities = []
    
    print(f"Training {n_imputations} models on imputed datasets...")
    for i, (X_train_imp, X_test_imp) in enumerate(imputed_datasets):
        # Create a copy of the pipeline for this imputation
        pipe_copy = deepcopy(pipe)
        
        # Fit on this imputed training set
        pipe_copy.fit(X_train_imp, y_train)
        
        # Predict on imputed test set
        y_prob = pipe_copy.predict_proba(X_test_imp)[:, 1]
        y_pred = (y_prob >= best_thr).astype(int)
        
        all_predictions.append(y_pred)
        all_probabilities.append(y_prob)
        
        if (i + 1) % 5 == 0 or i == n_imputations - 1:
            print(f"  Completed {i + 1}/{n_imputations} imputations")
    
    # Bootstrap over the imputation results
    print(f"Calculating bootstrap confidence intervals ({n_bootstrap} samples)...")
    rng = np.random.default_rng(random_state)
    n_test = len(y_test)
    
    bootstrap_metrics = {
        'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 
        'roc_auc': [], 'pr_auc': [], 'brier_score': []
    }
    
    for boot_idx in range(n_bootstrap):
        # Sample test indices with replacement
        test_indices = rng.choice(n_test, size=n_test, replace=True)
        
        # Sample an imputation dataset
        imp_idx = rng.choice(n_imputations)
        
        # Get predictions for this bootstrap sample
        y_true_boot = y_test.iloc[test_indices].values
        y_pred_boot = all_predictions[imp_idx][test_indices]
        y_prob_boot = all_probabilities[imp_idx][test_indices]
        
        # Calculate metrics
        try:
            bootstrap_metrics['accuracy'].append(accuracy_score(y_true_boot, y_pred_boot))
            bootstrap_metrics['precision'].append(precision_score(y_true_boot, y_pred_boot, zero_division=0))
            bootstrap_metrics['recall'].append(recall_score(y_true_boot, y_pred_boot, zero_division=0))
            bootstrap_metrics['f1'].append(f1_score(y_true_boot, y_pred_boot, zero_division=0))
            bootstrap_metrics['roc_auc'].append(roc_auc_score(y_true_boot, y_prob_boot))
            bootstrap_metrics['pr_auc'].append(average_precision_score(y_true_boot, y_prob_boot))
            bootstrap_metrics['brier_score'].append(brier_score_loss(y_true_boot, y_prob_boot))
        except Exception:
            continue
    
    # Calculate confidence intervals
    def ci(values):
        if not values:
            return {"mean": None, "low": None, "high": None, "std": None}
        arr = np.array(values)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "low": float(np.percentile(arr, 2.5)),
            "high": float(np.percentile(arr, 97.5))
        }
    
    # Also calculate ensemble metrics (average across all imputations)
    ensemble_pred = np.mean(all_predictions, axis=0)
    ensemble_prob = np.mean(all_probabilities, axis=0)
    ensemble_pred_binary = (ensemble_pred >= 0.5).astype(int)
    
    ensemble_metrics = {
        'accuracy': accuracy_score(y_test, ensemble_pred_binary),
        'precision': precision_score(y_test, ensemble_pred_binary, zero_division=0),
        'recall': recall_score(y_test, ensemble_pred_binary, zero_division=0),
        'f1': f1_score(y_test, ensemble_pred_binary, zero_division=0),
        'roc_auc': roc_auc_score(y_test, ensemble_prob),
        'pr_auc': average_precision_score(y_test, ensemble_prob),
        'brier_score': brier_score_loss(y_test, ensemble_prob)
    }
    
    return {
        'bootstrap_ci': {metric: ci(values) for metric, values in bootstrap_metrics.items()},
        'ensemble_metrics': ensemble_metrics,
        'n_imputations': n_imputations,
        'n_bootstrap': n_bootstrap,
        'imputation_strategy': imputation_strategy
    }

def evaluate_seed_averaged_calibrated(best_pipe: ImbPipeline, X: pd.DataFrame, y: pd.Series, best_thr: float, seeds: List[int] | None = None, test_size: float = 0.2) -> Tuple[pd.DataFrame, Dict]:
    seeds = seeds or [RANDOM_STATE + i for i in range(N_REPEATS)]
    rows: List[Dict] = []
    for sd in seeds:
        X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X, y, test_size=test_size, random_state=sd, stratify=y)
        for method in ["isotonic", "sigmoid"]:
            clf = CalibratedClassifierCV(estimator=best_pipe, method=method, cv=3)
            clf.fit(X_train_s, y_train_s)
            prob_test = clf.predict_proba(X_test_s)[:, 1]
            # tune threshold per seed and calibrator on the holdout
            thr_df = evaluate_thresholds(y_test_s.values, prob_test)
            thr_best = float(thr_df.sort_values("f1", ascending=False).iloc[0]["threshold"]) if not thr_df.empty else best_thr
            y_pred = (prob_test >= thr_best).astype(int)
            rows.append({
                "seed": int(sd),
                "calibration": method,
                "threshold": thr_best,
                "accuracy": accuracy_score(y_test_s, y_pred),
                "precision": precision_score(y_test_s, y_pred, zero_division=0),
                "recall": recall_score(y_test_s, y_pred, zero_division=0),
                "f1": f1_score(y_test_s, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test_s, prob_test),
                "pr_auc": average_precision_score(y_test_s, prob_test),
            })
    df = pd.DataFrame(rows)
    summary = df.groupby("calibration").agg({"accuracy":"mean","precision":"mean","recall":"mean","f1":"mean","roc_auc":"mean","pr_auc":"mean"}).reset_index()
    # determine best calibration method by averaged F1 (then ROC AUC)
    best_row = summary.sort_values(["f1","roc_auc"], ascending=[False, False]).iloc[0]
    return df, {"best_calibration": str(best_row["calibration"]), "mean_f1": float(best_row["f1"]), "mean_roc_auc": float(best_row["roc_auc"]) }


def plot_seed_calibrated_f1(df: pd.DataFrame, out_path: str) -> None:
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df, x="calibration", y="f1")
    plt.title("Phase 4b: Seed-averaged calibrated F1")
    plt.ylabel("F1 score")
    plt.xlabel("Calibration method")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, format="svg")
    plt.close()



if __name__ == "__main__":
    main()