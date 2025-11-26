import os
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone, BaseEstimator, TransformerMixin
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE, RandomOverSampler
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, NearMiss

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from joblib import dump


REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")
VISUALS_DIR = os.path.join(os.path.dirname(__file__), "visuals")
TEST_VIS_DIR = os.path.join(VISUALS_DIR, "test_cases")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

for d in [REPORTS_DIR, VISUALS_DIR, TEST_VIS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)


def detect_target_column(df: pd.DataFrame) -> str:
    # Prefer explicit death outcome style columns first (case/space-insensitive)
    for col in df.columns:
        name = str(col).strip().lower()
        if "death" in name and "outcome" in name:
            return col
    # Next, any column mentioning death
    for col in df.columns:
        name = str(col).strip().lower()
        if "death" in name and all(k not in name for k in ["timing", "therapy", "level", "type"]):
            return col
    # Then other common target names
    candidates = [
        "target", "label", "y", "stroke", "death", "DEATH", "outcome", "Outcome",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: last binary column
    for col in reversed(df.columns):
        s = df[col].dropna()
        # treat typical yes/no as binary
        uniq = set(str(v).strip().lower() for v in s.unique())
        if len(uniq) == 2 and uniq.issubset({"yes", "no", "1", "0", "true", "false"}):
            return col
        if s.dtype.kind in {"i", "u", "f"} and s.isin([0, 1]).all():
            return col
    # last column as ultimate fallback
    return df.columns[-1]


def coerce_binary_target(series: pd.Series) -> pd.Series:
    """Map common string/object binary labels to {0,1}. Leaves numeric as-is.
    Positive terms map to 1, others to 0 when there are exactly two unique values.
    """
    if series.dtype.kind in {"i", "u", "f"}:
        # Numeric: try to coerce to 0/1 directly
        return series.astype(float)
    # Object-like: try mapping
    non_null = series.dropna()
    uniq = sorted(set(str(v).strip().lower() for v in non_null.unique()))
    if len(uniq) == 2:
        pos_terms = {
            "1", "yes", "y", "true", "t", "positive", "pos", "death", "deceased", "stroke",
        }
        mapping = {}
        for v in series.unique():
            if pd.isna(v):
                mapping[v] = np.nan
            else:
                key = str(v).strip().lower()
                mapping[v] = 1.0 if key in pos_terms else 0.0
        return series.map(mapping)
    # Otherwise, return as-is (will likely be multi-class; metrics will degrade)
    return series

# Added: Optional Winsorizer (percentile clipper) for numeric features
class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, quantile_range: Tuple[float, float] = (0.01, 0.99)):
        self.quantile_range = quantile_range
        self.lower_: np.ndarray | None = None
        self.upper_: np.ndarray | None = None

    def fit(self, X, y=None):
        X_arr = X.values if hasattr(X, "values") else np.asarray(X)
        q_low, q_high = self.quantile_range
        self.lower_ = np.nanquantile(X_arr, q_low, axis=0)
        self.upper_ = np.nanquantile(X_arr, q_high, axis=0)
        return self

    def transform(self, X):
        X_is_df = hasattr(X, "values") and hasattr(X, "columns")
        X_arr = X.values if X_is_df else np.asarray(X)
        X_clip = np.clip(X_arr, self.lower_, self.upper_)
        if X_is_df:
            return pd.DataFrame(X_clip, index=X.index, columns=X.columns)
        return X_clip

# Added: Analysis helpers

def analyze_missingness(df: pd.DataFrame, reports_dir: str, visuals_dir: str, phase_tag: str = "phase2b") -> None:
    try:
        miss_counts = df.isna().sum()
        miss_pct = (miss_counts / len(df)).sort_values(ascending=False)
        miss_df = pd.DataFrame({"feature": miss_pct.index, "missing_fraction": miss_pct.values})
        miss_df.to_csv(os.path.join(reports_dir, f"{phase_tag}_missingness_summary.csv"), index=False)
        # Bar plot (top 30 non-zero)
        non_zero = miss_df[miss_df["missing_fraction"] > 0].head(30)
        if not non_zero.empty:
            plt.figure(figsize=(10, max(3, 0.35 * len(non_zero))))
            sns.barplot(y="feature", x="missing_fraction", data=non_zero, color="#4C78A8")
            plt.xlabel("Missing fraction")
            plt.ylabel("Feature")
            plt.title(f"Missingness (Top {len(non_zero)} features)")
            plt.tight_layout()
            plt.savefig(os.path.join(visuals_dir, "missingness_bar.svg"), format="svg")
            plt.close()
        # Missingness matrix (first 200 rows)
        try:
            sample = df.head(200)
            plt.figure(figsize=(min(12, 0.25 * sample.shape[1] + 3), 6))
            sns.heatmap(sample.isna(), cbar=False)
            plt.title("Missingness Matrix (first 200 rows)")
            plt.tight_layout()
            plt.savefig(os.path.join(visuals_dir, "missingness_matrix.svg"), format="svg")
            plt.close()
        except Exception:
            pass
    except Exception as e:
        print(f"Warning: missingness analysis failed: {e}")


def analyze_outliers(df_X: pd.DataFrame, numeric_cols: List[str], reports_dir: str, visuals_dir: str, phase_tag: str = "phase2b") -> None:
    try:
        if not numeric_cols:
            return
        stats = []
        Xnum = df_X[numeric_cols]
        Q1 = Xnum.quantile(0.25)
        Q3 = Xnum.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_rows = len(Xnum)
        for col in numeric_cols:
            col_series = Xnum[col]
            mask = (col_series < lower[col]) | (col_series > upper[col])
            cnt = int(mask.sum())
            stats.append({"feature": col, "outlier_count": cnt, "outlier_fraction": cnt / n_rows if n_rows else 0.0})
        out_df = pd.DataFrame(stats).sort_values("outlier_fraction", ascending=False)
        out_df.to_csv(os.path.join(reports_dir, f"{phase_tag}_outliers_iqr_summary.csv"), index=False)
        # Bar of top 20
        topk = out_df.head(20)
        if not topk.empty:
            plt.figure(figsize=(10, max(3, 0.35 * len(topk))))
            sns.barplot(y="feature", x="outlier_fraction", data=topk, color="#E45756")
            plt.xlabel("Outlier fraction (IQR rule)")
            plt.ylabel("Feature")
            plt.title("Outliers by Feature (Top 20)")
            plt.tight_layout()
            plt.savefig(os.path.join(visuals_dir, "outliers_iqr_bar.svg"), format="svg")
            plt.close()
        # Boxplots grid for top 9
        top9 = topk.head(9)["feature"].tolist()
        if top9:
            n = len(top9)
            rows = int(np.ceil(n / 3))
            fig, axes = plt.subplots(rows, 3, figsize=(12, 3.2 * rows))
            axes = np.atleast_1d(axes).flatten()
            for ax, col in zip(axes, top9):
                sns.boxplot(x=Xnum[col], ax=ax, color="#72B7B2")
                ax.set_title(col)
            for ax in axes[n:]:
                ax.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(visuals_dir, "outliers_boxplots.svg"), format="svg")
            plt.close()
    except Exception as e:
        print(f"Warning: outlier analysis failed: {e}")


def write_handling_strategies_report(numeric_cols: List[str], categorical_cols: List[str], outlier_treatment: str, reports_dir: str, phase_tag: str = "phase2b") -> None:
    try:
        payload = {
            "phase": phase_tag,
            "missing_values": {
                "numeric": {"strategy": "median", "affected_features": numeric_cols},
                "categorical": {"strategy": "most_frequent", "affected_features": categorical_cols}
            },
            "outliers": {
                "strategy": outlier_treatment,
                "notes": "winsorize clips extreme values to chosen quantile bounds; 'none' leaves data unchanged"
            }
        }
        with open(os.path.join(reports_dir, f"{phase_tag}_preprocessing_strategies.json"), "w") as f:
            json.dump(payload, f, indent=2)
    except Exception as e:
        print(f"Warning: failed to write strategies report: {e}")

# New: build_preprocessor with optional outlier winsorization for numeric features
class MissingnessIndicator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        import pandas as pd
        X_df = pd.DataFrame(X)
        return X_df.isna().astype(float).values

def build_preprocessor(df: pd.DataFrame, target_col: str) -> Tuple[ColumnTransformer, List[str], List[str]]:
    features = df.drop(columns=[target_col])
    numeric_features = features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in features.columns if c not in numeric_features]

    outlier_treatment = os.environ.get("PHASE2B_OUTLIER_TREATMENT", "none").lower()
    miss_mode = os.environ.get("PHASE2B_MISSINGNESS_MODE", "before").lower()

    # Numeric pipeline: BEFORE = SimpleImputer+Scaler, AFTER = KNNImputer+Scaler
    if miss_mode == "after":
        num_steps: List[Tuple[str, object]] = [("imputer", KNNImputer(n_neighbors=5, weights="distance"))]
    else:
        num_steps: List[Tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]

    if outlier_treatment in ("winsorize", "clip"):
        num_steps.append(("winsor", Winsorizer(quantile_range=(0.01, 0.99))))
    num_steps.append(("scaler", StandardScaler()))
    numeric_transformer = Pipeline(steps=num_steps)

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = [
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]

    # AFTER mode: add missingness indicator columns (dense) for both num and cat
    if miss_mode == "after":
        transformers.append(("num_miss", MissingnessIndicator(), numeric_features))
        transformers.append(("cat_miss", MissingnessIndicator(), categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers)
    return preprocessor, numeric_features, categorical_features


def model_specs(class_weight: str = None) -> Dict[str, object]:
    cw = class_weight if class_weight in ("balanced", None) else None
    specs: Dict[str, object] = {
        "lr": LogisticRegression(max_iter=2000, class_weight=cw, solver="lbfgs"),
        "rf": RandomForestClassifier(n_estimators=300, random_state=42, class_weight=cw),
        "gb": GradientBoostingClassifier(random_state=42),  # no class_weight support
        "svc": SVC(kernel="rbf", probability=True, class_weight=cw, random_state=42),
        "knn": KNeighborsClassifier(n_neighbors=15),
        "gnb": GaussianNB(),
        # Calibrated variants for improved probability estimates
        "lr_cal": CalibratedClassifierCV(LogisticRegression(max_iter=2000, class_weight=cw, solver="lbfgs"), method="sigmoid", cv=3),
        "gnb_cal": CalibratedClassifierCV(GaussianNB(), method="sigmoid", cv=3),
    }
    return specs

def get_sampler(name: str):
    name = (name or "none").lower()
    if name in ("none", "nosmote"):
        return None
    if name == "smote":
        return SMOTE(random_state=42)
    if name == "adasyn":
        return ADASYN(random_state=42)
    if name in ("bsmote", "borderlinesmote", "borderline"):
        return BorderlineSMOTE(random_state=42, kind="borderline-1")
    if name == "svmsmote":
        return SVMSMOTE(random_state=42)
    if name == "smotetomek":
        return SMOTETomek(random_state=42)
    if name == "smoteenn":
        return SMOTEENN(random_state=42)
    if name in ("ros", "randomover", "randomoversampler"):
        return RandomOverSampler(random_state=42)
    if name in ("rus", "randomunder", "randomundersampler"):
        return RandomUnderSampler(random_state=42)
    if name in ("tomek", "tomeklinks"):
        return TomekLinks()
    if name == "nearmiss":
        return NearMiss()
    return None



# ... existing code ...
# Added: FeatureBoundsClipper transformer and integrate into numeric preprocessing; add CV-based threshold selection; use locked threshold for predictions; minor imports if needed.
class FeatureBoundsClipper(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names: List[str], bounds: Dict[str, Tuple[float, float]] | None = None):
        self.feature_names = feature_names
        self.bounds = bounds or {}
        # Map index -> (min, max)
        self._idx_bounds: Dict[int, Tuple[float, float]] = {}

    def fit(self, X, y=None):
        # Build index mapping once using provided feature_names
        for i, name in enumerate(self.feature_names):
            if name in self.bounds:
                try:
                    lo, hi = self.bounds[name]
                    if lo is not None and hi is not None and float(lo) <= float(hi):
                        self._idx_bounds[i] = (float(lo), float(hi))
                except Exception:
                    continue
        return self

    def transform(self, X):
        import numpy as _np
        arr = X.values if hasattr(X, "values") else _np.asarray(X)
        if not self._idx_bounds:
            return arr
        # Copy to avoid mutating input
        out = arr.copy()
        for i, (lo, hi) in self._idx_bounds.items():
            out[:, i] = _np.clip(out[:, i], lo, hi)
        return out

# Duplicate build_preprocessor removed to preserve original implementation with winsorization and missingness handling.


def model_specs(class_weight: str = None) -> Dict[str, object]:
    cw = class_weight if class_weight in ("balanced", None) else None
    specs: Dict[str, object] = {
        "lr": LogisticRegression(max_iter=2000, class_weight=cw, solver="lbfgs"),
        "rf": RandomForestClassifier(n_estimators=300, random_state=42, class_weight=cw),
        "gb": GradientBoostingClassifier(random_state=42),  # no class_weight support
        "svc": SVC(kernel="rbf", probability=True, class_weight=cw, random_state=42),
        "knn": KNeighborsClassifier(n_neighbors=15),
        "gnb": GaussianNB(),
        # Calibrated variants for improved probability estimates
        "lr_cal": CalibratedClassifierCV(LogisticRegression(max_iter=2000, class_weight=cw, solver="lbfgs"), method="sigmoid", cv=3),
        "gnb_cal": CalibratedClassifierCV(GaussianNB(), method="sigmoid", cv=3),
    }
    return specs

def get_sampler(name: str):
    name = (name or "none").lower()
    if name in ("none", "nosmote"):
        return None
    if name == "smote":
        return SMOTE(random_state=42)
    if name == "adasyn":
        return ADASYN(random_state=42)
    if name in ("bsmote", "borderlinesmote", "borderline"):
        return BorderlineSMOTE(random_state=42, kind="borderline-1")
    if name == "svmsmote":
        return SVMSMOTE(random_state=42)
    if name == "smotetomek":
        return SMOTETomek(random_state=42)
    if name == "smoteenn":
        return SMOTEENN(random_state=42)
    if name in ("ros", "randomover", "randomoversampler"):
        return RandomOverSampler(random_state=42)
    if name in ("rus", "randomunder", "randomundersampler"):
        return RandomUnderSampler(random_state=42)
    if name in ("tomek", "tomeklinks"):
        return TomekLinks()
    if name == "nearmiss":
        return NearMiss()
    return None



# ... existing code ...
# CV-based threshold selection to improve F1 calibration
def _select_threshold_via_cv(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, n_splits: int = 5, thr_grid: np.ndarray | None = None) -> float:
    if thr_grid is None:
        thr_grid = np.linspace(0.3, 0.7, 17)
    best_thr, best_score = 0.5, -1.0
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for thr in thr_grid:
        scores = []
        for tr_idx, va_idx in skf.split(X, y):
            Xt, Xv = X.iloc[tr_idx], X.iloc[va_idx]
            yt, yv = y.iloc[tr_idx], y.iloc[va_idx]
            try:
                pipe.fit(Xt, yt)
                prob = pipe.predict_proba(Xv)[:, 1]
                pred = (prob >= thr).astype(int)
                scores.append(f1_score(yv, pred, zero_division=0))
            except Exception:
                continue
        if scores:
            m = float(np.mean(scores))
            if m > best_score:
                best_score, best_thr = m, float(thr)
    return float(best_thr)

def build_pipelines(preprocessor: ColumnTransformer, sampler_name: str, class_weight: str = None, y_train: pd.Series = None) -> Dict[str, Pipeline]:
    estimators = model_specs(class_weight)
    pipelines: Dict[str, Pipeline] = {}

    sampler = get_sampler(sampler_name)
    # Compute capped ratio for oversamplers when y_train provided
    ratio = _compute_sampling_ratio(y_train) if y_train is not None else None
    for name, est in estimators.items():
        preproc = clone(preprocessor)
        if sampler is None:
            pipelines[f"{name}_{'balanced' if class_weight else 'standard'}_none"] = Pipeline(steps=[("preprocess", preproc), ("clf", est)])
        else:
            sname = (sampler_name or "none").lower()
            sampler_inst = None
            try:
                if sname == "smote":
                    kwargs = {"random_state": 42}
                    if ratio is not None and ratio > 0:
                        kwargs["sampling_strategy"] = ratio
                    sampler_inst = SMOTE(**kwargs)
                elif sname == "adasyn":
                    kwargs = {"random_state": 42}
                    if ratio is not None and ratio > 0:
                        kwargs["sampling_strategy"] = ratio
                    sampler_inst = ADASYN(**kwargs)
                elif sname in ("bsmote", "borderlinesmote", "borderline"):
                    kwargs = {"random_state": 42, "kind": "borderline-1"}
                    if ratio is not None and ratio > 0:
                        kwargs["sampling_strategy"] = ratio
                    sampler_inst = BorderlineSMOTE(**kwargs)
                elif sname == "svmsmote":
                    kwargs = {"random_state": 42}
                    if ratio is not None and ratio > 0:
                        kwargs["sampling_strategy"] = ratio
                    sampler_inst = SVMSMOTE(**kwargs)
                elif sname == "smotetomek":
                    kwargs = {"random_state": 42}
                    if ratio is not None and ratio > 0:
                        kwargs["sampling_strategy"] = ratio
                    sampler_inst = SMOTETomek(**kwargs)
                elif sname == "smoteenn":
                    kwargs = {"random_state": 42}
                    if ratio is not None and ratio > 0:
                        kwargs["sampling_strategy"] = ratio
                    sampler_inst = SMOTEENN(**kwargs)
                elif sname in ("ros", "randomover", "randomoversampler"):
                    kwargs = {"random_state": 42}
                    if ratio is not None and ratio > 0:
                        kwargs["sampling_strategy"] = ratio
                    sampler_inst = RandomOverSampler(**kwargs)
                else:
                    sampler_inst = sampler
            except Exception:
                sampler_inst = sampler
            pipelines[f"{name}_{'balanced' if class_weight else 'standard'}_{sampler_name.lower()}"] = ImbPipeline(steps=[("preprocess", preproc), ("sampler", sampler_inst), ("clf", est)])
    return pipelines


def evaluate_model(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else float("nan"),
    }


def pick_test_cases(X: pd.DataFrame, y: pd.Series, n_per_class: int = 5) -> pd.DataFrame:
    # Build simple class centroids on numeric-only standardized space
    num_cols = X.select_dtypes(include=[np.number]).columns
    if len(num_cols) == 0:
        # If no numeric columns, just sample n_per_class from each label
        pos_idx = y[y == 1].sample(min(n_per_class, (y == 1).sum()), random_state=42).index
        neg_idx = y[y == 0].sample(min(n_per_class, (y == 0).sum()), random_state=42).index
        idx = list(pos_idx) + list(neg_idx)
        out = X.loc[idx].copy()
        out["__expected_label__"] = y.loc[idx].values
        return out

    X_num = X[num_cols].copy()
    X_num = (X_num - X_num.mean()) / (X_num.std() + 1e-9)

    cases_indices = []
    for label in [0, 1]:
        mask = (y == label)
        subset = X_num[mask]
        if subset.empty:
            continue
        centroid = subset.mean(axis=0)
        dists = ((subset - centroid) ** 2).sum(axis=1)
        top_idx = dists.nsmallest(n_per_class).index
        cases_indices.extend(list(top_idx))

    out = X.loc[cases_indices].copy()
    out["__expected_label__"] = y.loc[cases_indices].values
    return out


def plot_confusion_matrices(cm_map: Dict[str, np.ndarray], out_path: str) -> None:
    n = len(cm_map)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()
    for ax, (name, cm) in zip(axes, cm_map.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    for ax in axes[n:]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_roc_curves(roc_map: Dict[str, Tuple[np.ndarray, np.ndarray]], out_path: str) -> None:
    plt.figure(figsize=(8,6))
    for name, (fpr, tpr) in roc_map.items():
        plt.plot(fpr, tpr, label=name)
    plt.plot([0,1],[0,1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (Phase 2b)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def evaluate_test_cases(models: Dict[str, Pipeline], df: pd.DataFrame, target_col: str, top_k: int = 8) -> None:
    X = df.drop(columns=[target_col])
    y = df[target_col]
    cases = pick_test_cases(X, y, n_per_class=5)

    # Export test case parameters (feature values) to JSON
    try:
        params_payload = {
            "selection": {
                "method": "per-class centroid nearest",
                "n_per_class": 5
            },
            "test_cases": []
        }
        for i, idx in enumerate(cases.index):
            case_name = f"{'Negative' if cases.loc[idx, '__expected_label__']==0 else 'Death'} Case {i+1}"
            feat_row = cases.drop(columns=["__expected_label__"]).loc[idx].to_dict()
            # Convert numpy types to native Python for JSON serialization
            clean_feats = {}
            for k, v in feat_row.items():
                if pd.isna(v):
                    clean_feats[k] = None
                elif isinstance(v, (np.integer,)):
                    clean_feats[k] = int(v)
                elif isinstance(v, (np.floating,)):
                    clean_feats[k] = float(v)
                else:
                    clean_feats[k] = v
            params_payload["test_cases"].append({
                "case_id": int(idx),
                "case_name": case_name,
                "expected_label": int(cases.loc[idx, "__expected_label__"]),
                "features": clean_feats
            })
        with open(os.path.join(REPORTS_DIR, "phase2b_test_cases_parameters.json"), "w") as f:
            json.dump(params_payload, f, indent=2)
    except Exception as e:
        # Non-fatal: continue pipeline even if JSON export fails
        print(f"Warning: failed to write test case parameters JSON: {e}")

    summary_rows = []
    # rank models by f1 from saved metrics
    metrics_path = os.path.join(REPORTS_DIR, "phase2b_metrics.csv")
    if os.path.exists(metrics_path):
        metrics_df = pd.read_csv(metrics_path)
        # Ensure at least one top model per augmentation family appears
        def aug_family(mname: str) -> str:
            parts = str(mname).split("_")
            return parts[-1] if parts else "none"
        fam_best = (metrics_df.sort_values("f1", ascending=False)
                    .groupby(metrics_df["model"].apply(aug_family), as_index=False)
                    .first())
        fam_models = fam_best["model"].tolist()
        # If more than top_k families, truncate; else, fill with next best overall
        if len(fam_models) >= top_k:
            top_models = fam_models[:top_k]
        else:
            remaining = metrics_df[~metrics_df["model"].isin(fam_models)]
            extra = remaining.sort_values("f1", ascending=False)["model"].tolist()
            top_models = fam_models + extra[: max(0, top_k - len(fam_models))]
    else:
        top_models = list(models.keys())[:top_k]

    results_csv = os.path.join(REPORTS_DIR, "phase2b_test_cases_results.csv")
    rows = []
    for name in top_models:
        model = models[name]
        probs = model.predict_proba(cases.drop(columns=["__expected_label__"]))[:, 1]
        preds = (probs >= 0.5).astype(int)
        for i, idx in enumerate(cases.index):
            case_name = f"{'Negative' if cases.loc[idx, '__expected_label__']==0 else 'Death'} Case {i+1}"
            rows.append({
                "model": name,
                "case_id": int(idx),
                "case_name": case_name,
                "expected_label": int(cases.loc[idx, "__expected_label__"]),
                "pred_prob": float(probs[i]),
                "pred_label": int(preds[i]),
            })
    pd.DataFrame(rows).to_csv(results_csv, index=False)

    # barplot
    df_plot = pd.DataFrame(rows)
    plt.figure(figsize=(12,6))
    sns.barplot(data=df_plot, x="case_name", y="pred_prob", hue="model")
    ax = plt.gca()
    # add value labels on bars
    for p in ax.patches:
        h = p.get_height()
        if not np.isnan(h):
            ax.annotate(f"{h:.2f}", (p.get_x() + p.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7, rotation=0)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Predicted probability (positive)")
    plt.title("Per Test Case Comparison (Top Models incl. Augmentations)")
    plt.tight_layout()
    plt.savefig(os.path.join(TEST_VIS_DIR, "per_test_case_comparison.png"), dpi=150)
    plt.close()
    # Balanced-only per-test-case comparison
    try:
        df_bal = df_plot[df_plot["model"].astype(str).str.contains("_balanced_")]
        if not df_bal.empty:
            plt.figure(figsize=(12,6))
            sns.barplot(data=df_bal, x="case_name", y="pred_prob", hue="model")
            ax = plt.gca()
            for p in ax.patches:
                h = p.get_height()
                if not np.isnan(h):
                    ax.annotate(f"{h:.2f}", (p.get_x() + p.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7)
            plt.xticks(rotation=35, ha="right")
            plt.ylabel("Predicted probability (positive)")
            plt.title("Per Test Case Comparison (Balanced Only)")
            plt.tight_layout()
            plt.savefig(os.path.join(TEST_VIS_DIR, "per_test_case_comparison_balanced.png"), dpi=150)
            plt.close()
    except Exception:
        pass


def plot_metrics_table(metrics_df: pd.DataFrame, out_path: str, title: str = "Model Comparison (Phase 2b)") -> None:
    if metrics_df is None or metrics_df.empty:
        return
    df = metrics_df.copy()
    keep_cols = ["model", "accuracy", "precision", "recall", "f1", "roc_auc"]
    df = df[keep_cols]
    num_cols = [c for c in keep_cols if c != "model"]
    df[num_cols] = df[num_cols].astype(float).round(3)
    df = df.sort_values("f1", ascending=False)

    # mark top model and identify balanced rows
    if not df.empty:
        df.iloc[0, df.columns.get_loc("model")] = "★ " + str(df.iloc[0]["model"])  # add star to best F1
    balanced_mask = df["model"].astype(str).str.contains("_balanced_")

    n_rows = len(df) + 1
    fig_height = max(4, 0.35 * n_rows)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis('off')
    ax.set_title(title, fontsize=12, pad=12)

    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     cellLoc='center',
                     colLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.25)

    # highlight balanced rows (light orange)
    for i in range(len(df)):
        if balanced_mask.iloc[i]:
            for j in range(len(df.columns)):
                table[(i+1, j)].set_facecolor('#FFF4E6')
                table[(i+1, j)].set_edgecolor('#E6A96B')

    # add footnote
    fig.text(0.5, 0.02, '★ denotes best overall F1. Balanced models highlighted in light orange.', ha='center', va='center', fontsize=9)

    plt.tight_layout(rect=(0, 0.04, 1, 1))
    plt.savefig(out_path, format="svg")
    plt.close()


def main():
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "clean_data.csv")
    df = pd.read_csv(data_path)

    target_col = detect_target_column(df)
    # Drop rows with missing target and coerce to binary when applicable
    df = df.dropna(subset=[target_col]).copy()
    df[target_col] = coerce_binary_target(df[target_col])
    df = df.dropna(subset=[target_col]).copy()

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(float)

    # New: analyses for report Section 3.2 and 3.3
    try:
        analyze_missingness(df, REPORTS_DIR, VISUALS_DIR, phase_tag="phase2b")
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in X.columns if c not in num_cols]
        analyze_outliers(X, num_cols, REPORTS_DIR, VISUALS_DIR, phase_tag="phase2b")
        write_handling_strategies_report(num_cols, cat_cols, os.environ.get("PHASE2B_OUTLIER_TREATMENT", "none"), REPORTS_DIR, phase_tag="phase2b")
    except Exception as e:
        print(f"Warning: analysis steps failed: {e}")

    preprocessor, _, _ = build_preprocessor(df, target_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) == 2 else None
    )

    # Build variants across multiple augmentation samplers
    sampler_names = ["none", "smote", "adasyn", "bsmote", "svmsmote", "smotetomek", "smoteenn"]
    configs = [(sampler_name, cw) for sampler_name in sampler_names for cw in [None, "balanced"]]

    all_models: Dict[str, Pipeline] = {}
    metrics_rows = []
    cms: Dict[str, np.ndarray] = {}
    rocs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    from sklearn.metrics import roc_curve

    for sampler_name, cw in configs:
        models = build_pipelines(preprocessor, sampler_name=sampler_name, class_weight=cw)
        for name, pipe in models.items():
            label = f"{name}"
            if sampler_name != "none" and len(np.unique(y_train)) < 2:
                continue
            # lock threshold via CV on training set
            locked_thr = _select_threshold_via_cv(pipe, X_train, y_train, n_splits=5)
            try:
                pipe.fit(X_train, y_train)
            except Exception:
                continue
            all_models[label] = pipe
            safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in label)
            dump(pipe, os.path.join(MODELS_DIR, f"{safe_name}.joblib"), compress=3)
            # Test metrics using locked threshold
            prob = pipe.predict_proba(X_test)[:, 1]
            pred = (prob >= locked_thr).astype(int)
            m = evaluate_model(y_test, prob, pred)
            # Train metrics
            try:
                prob_tr = pipe.predict_proba(X_train)[:, 1]
                pred_tr = (prob_tr >= locked_thr).astype(int)
                train_m = evaluate_model(y_train, prob_tr, pred_tr)
            except Exception:
                train_m = {"accuracy": np.nan, "precision": np.nan, "recall": np.nan, "f1": np.nan, "roc_auc": np.nan}
            # CV F1 for generalization gap
            cv_mean, cv_std = np.nan, np.nan
            try:
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="f1")
                cv_mean, cv_std = float(scores.mean()), float(scores.std())
            except Exception:
                pass
            gen_gap = float(cv_mean) - float(m.get("f1", np.nan)) if not np.isnan(cv_mean) and not np.isnan(m.get("f1", np.nan)) else float("nan")
            m_row = {"model": label, **m,
                     "train_accuracy": train_m.get("accuracy"),
                     "train_precision": train_m.get("precision"),
                     "train_recall": train_m.get("recall"),
                     "train_f1": train_m.get("f1"),
                     "train_roc_auc": train_m.get("roc_auc"),
                     "cv_f1_mean": cv_mean, "cv_f1_std": cv_std, "generalization_gap": gen_gap,
                     "thr_locked": float(locked_thr)}
            metrics_rows.append(m_row)
            cms[label] = confusion_matrix(y_test, pred)
            try:
                fpr, tpr, _ = roc_curve(y_test, prob)
                rocs[label] = (fpr, tpr)
            except Exception:
                pass

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_csv = os.path.join(REPORTS_DIR, "phase2b_metrics.csv")
    metrics_json = os.path.join(REPORTS_DIR, "phase2b_metrics.json")
    metrics_df.to_csv(metrics_csv, index=False)
    with open(metrics_json, "w") as f:
        json.dump({r["model"]: {k: v for k, v in r.items() if k != "model"} for r in metrics_rows}, f, indent=2)
    # Save a summary table: best model per augmentation family by F1
    if not metrics_df.empty:
        def _aug(m):
            parts = str(m).split("_")
            return parts[-1] if parts else "none"
        metrics_df["augmentation"] = metrics_df["model"].apply(_aug)
        best_by_aug = (metrics_df.sort_values("f1", ascending=False)
                                  .groupby("augmentation", as_index=False)
                                  .first())
        best_by_aug.to_csv(os.path.join(REPORTS_DIR, "phase2b_metrics_by_augmentation.csv"), index=False)

    # NEW: model comparison table under visuals
    plot_metrics_table(metrics_df, os.path.join(VISUALS_DIR, "model_comparison_table.svg"), title="Model Comparison (Phase 2b)")
    # Balanced-only comparison table
    try:
        df_bal = metrics_df[metrics_df["model"].astype(str).str.contains("_balanced_")]
        if not df_bal.empty:
            plot_metrics_table(df_bal, os.path.join(VISUALS_DIR, "model_comparison_table_balanced.svg"), title="Model Comparison (Balanced Only)")
    except Exception:
        pass

    plot_confusion_matrices(cms, os.path.join(VISUALS_DIR, "confusion_matrices.png"))
    if rocs:
        plot_roc_curves(rocs, os.path.join(VISUALS_DIR, "roc_curve.png"))

    # Evaluate top models on comprehensive test cases sampled from data
    evaluate_test_cases(all_models, df, target_col, top_k=16)

    print(f"Phase 2b completed. Metrics -> {metrics_csv}")


if __name__ == "__main__":
    main()