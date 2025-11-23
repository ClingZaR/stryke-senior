import os
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, average_precision_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE, RandomOverSampler
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, NearMiss

# Optional gradient boosting libraries
try:
    from xgboost import XGBClassifier  # type: ignore
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from catboost import CatBoostClassifier  # type: ignore
    HAS_CAT = True
except Exception:
    HAS_CAT = False

# Optional extras
try:
    import shap  # type: ignore
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

try:
    import optuna  # type: ignore
    HAS_OPTUNA = True
except Exception:
    HAS_OPTUNA = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump


REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")
VISUALS_DIR = os.path.join(os.path.dirname(__file__), "visuals")
TEST_VIS_DIR = os.path.join(VISUALS_DIR, "test_cases")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# Explicit selection policy for test-case comparisons
# Ensure each base model appears with these sampler variants (standard only) in the per-test-case plot
TESTCASE_SAMPLERS_PER_MODEL = ["smote", "adasyn", "smoteenn", "smotetomek", "svmsmote"]
INCLUDE_BALANCED_FOR_TESTCASE = True  # enable balanced variants in test-case figure
INCLUDE_ENSEMBLES_FIRST = True

# New: Global toggles
IMBALANCE_MODE = os.environ.get("PHASE3B_IMBALANCE_MODE", "augmentation").lower()  # one of: none, class_weight, augmentation
CALIBRATE = os.environ.get("PHASE3B_CALIBRATE", "true").lower() in ("1", "true", "yes")
CALIBRATION_METHOD = os.environ.get("PHASE3B_CALIB_METHOD", "isotonic").lower()  # isotonic or sigmoid
NESTED_CV_SPLITS = int(os.environ.get("PHASE3B_NESTED_CV_SPLITS", "5"))
# New toggles
HPO_ENABLED = os.environ.get("PHASE3B_HPO", "0").lower() in ("1", "true", "yes")
HPO_TRIALS = int(os.environ.get("PHASE3B_HPO_TRIALS", "30"))
SHAP_ENABLED = os.environ.get("PHASE3B_SHAP", "1").lower() in ("1", "true", "yes")
MAX_VIS_MODELS = int(os.environ.get("PHASE3B_MAX_VIS_MODELS", "8"))

for d in [REPORTS_DIR, VISUALS_DIR, TEST_VIS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)


def detect_target_column(df: pd.DataFrame) -> str:
    # Prefer explicit death outcome style columns first
    for col in df.columns:
        name = str(col).strip().lower()
        if "death" in name and "outcome" in name:
            return col
    for col in df.columns:
        name = str(col).strip().lower()
        if "death" in name and all(k not in name for k in ["timing", "therapy", "level", "type"]):
            return col
    candidates = [
        "target", "label", "y", "stroke", "death", "DEATH", "outcome", "Outcome",
    ]
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


# Added: Optional Winsorizer
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

# Added: analysis helpers

def analyze_missingness(df: pd.DataFrame, reports_dir: str, visuals_dir: str, phase_tag: str = "phase3b") -> None:
    try:
        miss_counts = df.isna().sum()
        miss_pct = (miss_counts / len(df)).sort_values(ascending=False)
        miss_df = pd.DataFrame({"feature": miss_pct.index, "missing_fraction": miss_pct.values})
        miss_df.to_csv(os.path.join(reports_dir, f"{phase_tag}_missingness_summary.csv"), index=False)
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


def analyze_outliers(df_X: pd.DataFrame, numeric_cols: List[str], reports_dir: str, visuals_dir: str, phase_tag: str = "phase3b") -> None:
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


def write_handling_strategies_report(numeric_cols: List[str], categorical_cols: List[str], outlier_treatment: str, reports_dir: str, phase_tag: str = "phase3b") -> None:
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


class MissingnessIndicator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        import pandas as pd
        return pd.DataFrame(X).isna().astype(float).values

# Added: FeatureBoundsClipper to optionally clip numeric features to predefined bounds
class FeatureBoundsClipper(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names: List[str], bounds: Dict[str, Tuple[float, float]] | None = None):
        self.feature_names = feature_names
        self.bounds = bounds or {}
        self._idx_bounds: Dict[int, Tuple[float, float]] = {}

    def fit(self, X, y=None):
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
        out = arr.copy()
        for i, (lo, hi) in self._idx_bounds.items():
            out[:, i] = _np.clip(out[:, i], lo, hi)
        return out

def build_preprocessor(df: pd.DataFrame, target_col: str) -> ColumnTransformer:
    features = df.drop(columns=[target_col])
    numeric_features = features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in features.columns if c not in numeric_features]

    outlier_treatment = os.environ.get("PHASE3B_OUTLIER_TREATMENT", "none").lower()
    miss_mode = os.environ.get("PHASE3B_MISSINGNESS_MODE", "before").lower()
    bounds_path = os.environ.get("PHASE3B_BOUNDS_PATH", "")
    bounds_clip_enabled = os.environ.get("PHASE3B_BOUNDS_CLIP", "false").lower() in ("1", "true", "yes")
    bounds: Dict[str, Tuple[float, float]] = {}
    if bounds_clip_enabled and bounds_path and os.path.exists(bounds_path):
        try:
            with open(bounds_path, "r") as f:
                bounds = json.load(f)
        except Exception as e:
            print(f"Warning: failed to load bounds file '{bounds_path}': {e}")

    # Numeric
    if miss_mode == "after":
        num_steps: List[Tuple[str, object]] = [("imputer", KNNImputer(n_neighbors=5, weights="distance"))]
    else:
        num_steps: List[Tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if bounds_clip_enabled and bounds:
        num_steps.append(("clipper", FeatureBoundsClipper(feature_names=numeric_features, bounds=bounds)))
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

    if miss_mode == "after":
        transformers.append(("num_miss", MissingnessIndicator(), numeric_features))
        transformers.append(("cat_miss", MissingnessIndicator(), categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers)
    setattr(preprocessor, "_numeric_features", numeric_features)
    setattr(preprocessor, "_categorical_features", categorical_features)
    return preprocessor


def base_models(cw=None) -> Dict[str, object]:
    cw = cw if cw in ("balanced", None) else None
    models: Dict[str, object] = {
        "lr": LogisticRegression(max_iter=2000, class_weight=cw, solver="lbfgs"),
        "rf": RandomForestClassifier(n_estimators=300, random_state=42, class_weight=cw),
        "gb": GradientBoostingClassifier(random_state=42),
        "svc": SVC(kernel="rbf", probability=True, class_weight=cw, random_state=42),
        "knn": KNeighborsClassifier(n_neighbors=15),
        "gnb": GaussianNB(),
    }
    # Add XGBoost if available
    if HAS_XGB:
        models["xgb"] = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
            use_label_encoder=False
        )
    # Add CatBoost if available
    if HAS_CAT:
        models["cat"] = CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.1,
            loss_function="Logloss",
            random_state=42,
            verbose=False,
            allow_writing_files=False
        )
    return models


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
    if name == "smotenc":
        return "SMOTENC"  # sentinel handled in pipeline construction
    if name in ("ros", "randomover", "randomoversampler"):
        return RandomOverSampler(random_state=42)
    if name in ("rus", "randomunder", "randomundersampler"):
        return RandomUnderSampler(random_state=42)
    if name in ("tomek", "tomeklinks"):
        return TomekLinks()
    if name == "nearmiss":
        return NearMiss()
    return None


def _compute_sampling_ratio(y):
    try:
        # y is expected to be a Pandas Series of binary labels 0/1
        counts = y.value_counts()
        n0 = int(counts.get(0, 0))
        n1 = int(counts.get(1, 0))
        if n0 == 0 or n1 == 0:
            return None
        n_min = min(n0, n1)
        n_maj = max(n0, n1)
        base_ratio = n_min / max(1, n_maj)
        cap_mult = float(os.environ.get("PHASE3B_SAMPLING_CAP_MULT", "2.0"))
        target_ratio = base_ratio * cap_mult
        return float(min(1.0, max(0.0, target_ratio)))
    except Exception:
        return None

def _maybe_calibrated_estimator(est):
    if not CALIBRATE:
        return est
    try:
        return CalibratedClassifierCV(estimator=est, cv=3, method=CALIBRATION_METHOD if CALIBRATION_METHOD in ("isotonic", "sigmoid") else "sigmoid")
    except TypeError:
        # for older sklearn versions
        return CalibratedClassifierCV(base_estimator=est, cv=3, method=CALIBRATION_METHOD if CALIBRATION_METHOD in ("isotonic", "sigmoid") else "sigmoid")


def build_single_pipelines(preprocessor: ColumnTransformer, sampler_name: str, cw=None, y_train=None) -> Dict[str, Pipeline]:
    models = base_models(cw)
    pipes: Dict[str, Pipeline] = {}

    # Special handling for SMOTENC: sampler before preprocessing with ordinal-encoded cats, then post OHE/scale
    if (sampler_name or "").lower() == "smotenc":
        num = getattr(preprocessor, "_numeric_features", [])
        cat = getattr(preprocessor, "_categorical_features", [])
        n_num = len(num)
        n_cat = len(cat)
        cat_idx = list(range(n_num, n_num + n_cat))
        # Pre-SMOTE: impute + ordinal-encode categorical to single columns
        pre_smote = ColumnTransformer(
            transformers=[
                ("num", SimpleImputer(strategy="median"), num),
                ("cat", Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                ]), cat),
            ]
        )
        # Post-SMOTE: scale numeric, one-hot encode categorical (now integer-coded)
        post_smote = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), list(range(0, n_num))),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_idx),
            ]
        )
        for name, est in models.items():
            calibrated_est = _maybe_calibrated_estimator(est)
            # Compute capped ratio for SMOTENC as well
            ratio = _compute_sampling_ratio(y_train) if y_train is not None else None
            # Build kwargs for SMOTENC including sampling_strategy when available
            _kwargs = {"categorical_features": cat_idx, "random_state": 42}
            if ratio is not None and ratio > 0:
                _kwargs["sampling_strategy"] = ratio
            pipes[f"{name}_{'balanced' if cw else 'standard'}_smotenc"] = ImbPipeline(steps=[
                ("pre_smote", pre_smote),
                ("sampler", __import__('imblearn').over_sampling.SMOTENC(**_kwargs)),
                ("post", post_smote),
                ("clf", calibrated_est),
            ])
        return pipes

    # Default path for other samplers
    sampler = get_sampler(sampler_name)
    # Compute capped ratio for oversamplers if y_train provided
    ratio = _compute_sampling_ratio(y_train) if y_train is not None else None
    for name, est in models.items():
        calibrated_est = _maybe_calibrated_estimator(est)
        if sampler is None:
            pipes[f"{name}_{'balanced' if cw else 'standard'}_none"] = Pipeline(
                steps=[("preprocess", preprocessor), ("clf", calibrated_est)]
            )
        else:
            # Recreate sampler with capped sampling_strategy when supported
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
            pipes[f"{name}_{'balanced' if cw else 'standard'}_{sampler_name}"] = ImbPipeline(
                steps=[("preprocess", preprocessor), ("sampler", sampler_inst), ("clf", calibrated_est)]
            )
    return pipes


def fit_and_score(pipe: Pipeline, X_train, y_train, X_test, y_test, threshold: float = 0.5) -> Dict[str, float]:
    pipe.fit(X_train, y_train)
    # Test metrics
    prob = pipe.predict_proba(X_test)[:, 1]
    pred = (prob >= float(threshold)).astype(int)
    # Train metrics (for overfitting diagnostics)
    try:
        prob_tr = pipe.predict_proba(X_train)[:, 1]
        pred_tr = (prob_tr >= float(threshold)).astype(int)
        train_metrics = {
            "train_accuracy": float(accuracy_score(y_train, pred_tr)),
            "train_precision": float(precision_score(y_train, pred_tr, zero_division=0)),
            "train_recall": float(recall_score(y_train, pred_tr, zero_division=0)),
            "train_f1": float(f1_score(y_train, pred_tr, zero_division=0)),
            "train_roc_auc": float(roc_auc_score(y_train, prob_tr)),
        }
    except Exception:
        train_metrics = {
            "train_accuracy": float("nan"),
            "train_precision": float("nan"),
            "train_recall": float("nan"),
            "train_f1": float("nan"),
            "train_roc_auc": float("nan"),
        }
    out = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, prob)),
    }
    out.update(train_metrics)
    return out


def _select_threshold_via_cv(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, n_splits: int = 5, thr_grid: np.ndarray = None) -> float:
    if thr_grid is None:
        thr_grid = np.linspace(0.1, 0.9, 17)
    thrs: List[float] = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for tr_idx, va_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        try:
            model = clone(pipe)
            model.fit(X_tr, y_tr)
            proba = model.predict_proba(X_va)[:, 1]
            best_t, best_f = 0.5, -1.0
            for t in thr_grid:
                preds = (proba >= t).astype(int)
                f = f1_score(y_va, preds, zero_division=0)
                if f > best_f:
                    best_f = f
                    best_t = float(t)
            thrs.append(best_t)
        except Exception:
            continue
    return float(np.median(thrs)) if thrs else 0.5


def evaluate_test_cases(models: Dict[str, Pipeline], df: pd.DataFrame, target_col: str, top_k: int = 8) -> None:
    X = df.drop(columns=[target_col])
    y = df[target_col]

    num_cols = X.select_dtypes(include=[np.number]).columns
    if len(num_cols) == 0:
        pos_idx = y[y == 1].sample(min(5, (y == 1).sum()), random_state=42).index
        neg_idx = y[y == 0].sample(min(5, (y == 0).sum()), random_state=42).index
        idx = list(pos_idx) + list(neg_idx)
        cases = X.loc[idx].copy()
        cases["__expected_label__"] = y.loc[idx].values
        selection_info = {"method": "random per-class sampling", "seed": 42, "attempted_per_class": 5}
    else:
        X_num = X[num_cols].copy()
        X_num = (X_num - X_num.mean()) / (X_num.std() + 1e-9)
        indices = []
        for label in [0, 1]:
            mask = (y == label)
            subset = X_num[mask]
            if subset.empty:
                continue
            centroid = subset.mean(axis=0)
            dists = ((subset - centroid) ** 2).sum(axis=1)
            top_idx = dists.nsmallest(5).index
            indices.extend(list(top_idx))
        cases = X.loc[indices].copy()
        cases["__expected_label__"] = y.loc[indices].values
        selection_info = {"method": "per-class centroid nearest", "n_per_class": 5}

    # Export test case parameters (feature values) to JSON
    try:
        params_payload = {
            "selection": selection_info,
            "test_cases": []
        }
        for i, idx in enumerate(cases.index):
            case_name = f"{'Negative' if cases.loc[idx, '__expected_label__']==0 else 'Death'} Case {i+1}"
            feat_row = cases.drop(columns=["__expected_label__"]).loc[idx].to_dict()
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
        with open(os.path.join(REPORTS_DIR, "phase3b_test_cases_parameters.json"), "w") as f:
            json.dump(params_payload, f, indent=2)
    except Exception as e:
        print(f"Warning: failed to write test case parameters JSON: {e}")

    metrics_path = os.path.join(REPORTS_DIR, "phase3b_metrics.csv")

    # Build an ordered selection list that ensures per-model sampler coverage
    curated: List[str] = []

    # 1) Ensembles first (optional)
    if INCLUDE_ENSEMBLES_FIRST:
        for ens in ["voting_soft", "stacking_lr"]:
            if ens in models:
                curated.append(ens)

    # 2) For each base model, include specified sampler variants (standard or balanced)
    # Detect which base models are present from the trained model keys
    def parse_parts(mname: str):
        # Expected: base_weight_sampler, e.g., 'xgb_standard_smote'
        parts = mname.split("_")
        base = parts[0] if len(parts) > 0 else mname
        weight = parts[1] if len(parts) > 1 else "standard"
        sampler = parts[2] if len(parts) > 2 else "none"
        return base, weight, sampler

    available_names = set(models.keys())
    # Heuristic set of bases, filtered by what exists
    candidate_bases = ["lr", "rf", "gb", "svc", "knn", "gnb", "xgb", "cat"]
    present_bases = []
    for b in candidate_bases:
        if any(name.startswith(b + "_") for name in available_names) or b in available_names:
            present_bases.append(b)

    weights_to_try = ["balanced", "standard"] if INCLUDE_BALANCED_FOR_TESTCASE else ["standard"]

    for base in present_bases:
        for weight in weights_to_try:
            for sampler in TESTCASE_SAMPLERS_PER_MODEL:
                candidate = f"{base}_{weight}_{sampler}"
                if candidate in available_names and candidate not in curated:
                    curated.append(candidate)

    # Fallback to metrics-based ordering if curated is too small or empty
    if (not curated) and os.path.exists(metrics_path):
        metrics_df = pd.read_csv(metrics_path)
        def aug_family(mname: str) -> str:
            parts = str(mname).split("_")
            return parts[-1] if parts else "none"
        fam_best = (
            metrics_df.sort_values("f1", ascending=False)
            .groupby(metrics_df["model"].apply(aug_family), as_index=False)
            .first()
        )
        fam_models = fam_best["model"].tolist()
        curated = fam_models

    # Truncate if a top_k was explicitly requested smaller than curated
    top_models = curated if top_k is None or len(curated) <= top_k else curated[:top_k]

    # Persist every model used in test cases
    for name in top_models:
        if name not in models:
            continue
        pipe = models[name]
        safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)
        dump(pipe, os.path.join(MODELS_DIR, f"{safe_name}.joblib"), compress=3)

    rows = []
    for name in top_models:
        if name not in models:
            continue
        pipe = models[name]
        probs = pipe.predict_proba(cases.drop(columns=["__expected_label__"]))[:, 1]
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

    out_csv = os.path.join(REPORTS_DIR, "phase3b_test_cases_results.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    plt.figure(figsize=(14,7))
    df_plot = pd.DataFrame(rows)
    sns.barplot(data=df_plot, x="case_name", y="pred_prob", hue="model")
    ax = plt.gca()
    for p in ax.patches:
        h = p.get_height()
        if not np.isnan(h):
            ax.annotate(f"{h:.2f}", (p.get_x() + p.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Predicted probability (positive)")
    plt.title("Per Test Case Comparison (Per-Model Sampler Coverage)")
    plt.tight_layout()
    plt.savefig(os.path.join(TEST_VIS_DIR, "per_test_case_comparison.png"), dpi=150)
    plt.close()
    # Balanced-only per-test-case comparison
    try:
        df_bal = df_plot[df_plot["model"].astype(str).str.contains("_balanced_")]
        if not df_bal.empty:
            plt.figure(figsize=(14,7))
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


def plot_confusion_matrices(cm_map: Dict[str, np.ndarray], out_path: str) -> None:
    if not cm_map:
        return
    # Sort by estimated performance (F1 if binary, else accuracy) and limit to top models
    def _score(cm: np.ndarray) -> float:
        total = cm.sum()
        acc = float(np.trace(cm) / total) if total > 0 else 0.0
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            denom = (2 * tp + fp + fn)
            f1 = float(2 * tp / denom) if denom > 0 else 0.0
            return f1
        return acc

    items = sorted(cm_map.items(), key=lambda kv: _score(kv[1]), reverse=True)[:MAX_VIS_MODELS]
    n = len(items)
    cols = 2 if n <= 4 else 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.5 * rows))
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for idx, ax in enumerate(axes):
        if idx >= n:
            ax.axis("off")
            continue
        name, cm = items[idx]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax, linewidths=0.5, linecolor="#EEEEEE")
        total = cm.sum()
        acc = float(np.trace(cm) / total) if total > 0 else 0.0
        subtitle = f"Acc={acc:.3f}"
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            denom = (2 * tp + fp + fn)
            f1 = float(2 * tp / denom) if denom > 0 else 0.0
            subtitle = f"Acc={acc:.3f} • F1={f1:.3f}"
        ax.set_title(f"{name}\n{subtitle}", fontsize=10)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("True", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, format="svg")
    plt.close()


def plot_roc_curves(roc_map: Dict[str, Tuple[np.ndarray, np.ndarray]], out_path: str, title: str = "ROC Curves (Phase 3b)") -> None:
    if not roc_map:
        return
    # Compute approximate AUC from the curve and sort; then limit to top models
    items = []
    for name, (fpr, tpr) in roc_map.items():
        try:
            auc = float(np.trapz(tpr, fpr))
        except Exception:
            auc = 0.0
        items.append((name, fpr, tpr, auc))
    items.sort(key=lambda x: x[3], reverse=True)
    items = items[:MAX_VIS_MODELS]

    plt.figure(figsize=(8.5, 6.5))
    colors = sns.color_palette("colorblind", n_colors=len(items))
    for (name, fpr, tpr, auc), color in zip(items, colors):
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", lw=2, color=color)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=1, label="Chance")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"Top {len(items)} ROC Curves")
    plt.legend(fontsize=8, loc='lower right', frameon=True)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, format="svg")
    plt.close()


def plot_pr_curves(pr_map: Dict[str, Tuple[np.ndarray, np.ndarray, float]], out_path: str, title: str = "Precision-Recall Curves (Phase 3b)") -> None:
    if not pr_map:
        return
    # Sort by AP and limit to top models
    items = sorted(pr_map.items(), key=lambda kv: kv[1][2], reverse=True)[:MAX_VIS_MODELS]

    plt.figure(figsize=(8.5, 6.5))
    colors = sns.color_palette("colorblind", n_colors=len(items))
    for (name, (prec, rec, ap)), color in zip(items, colors):
        plt.plot(rec, prec, label=f"{name} (AP={ap:.3f})", lw=2, color=color)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Top {len(items)} Precision-Recall Curves")
    plt.legend(fontsize=8, loc='lower left', frameon=True)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, format="svg")
    plt.close()


def threshold_sweep(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, thresholds: np.ndarray) -> Tuple[float, float, List[Dict[str, float]]]:
    # Returns best_threshold, best_f1, and full sweep rows
    prob = pipe.predict_proba(X_test)[:, 1]
    rows = []
    best_f1 = -1.0
    best_thr = 0.5
    for thr in thresholds:
        pred = (prob >= thr).astype(int)
        f1 = float(f1_score(y_test, pred, zero_division=0))
        rows.append({"threshold": float(thr), "f1": f1})
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr, best_f1, rows


def bootstrap_f1(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, n_boot: int = 200, random_state: int = 42) -> Tuple[float, float, float]:
    rng = np.random.default_rng(random_state)
    n = len(y_test)
    scores = []
    prob = pipe.predict_proba(X_test)[:, 1]
    # fix threshold at 0.5 for bootstrap to measure baseline classifier behavior
    pred_base = (prob >= 0.5).astype(int)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b = y_test.iloc[idx].values
        p_b = pred_base[idx]
        scores.append(float(f1_score(y_b, p_b, zero_division=0)))
    scores = np.array(scores)
    low, high = float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))
    return float(np.mean(scores)), low, high


def plot_metrics_table(metrics_df: pd.DataFrame, out_path: str, title: str = "Model Comparison Table") -> None:
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

    # highlight balanced rows (light orange overlay)
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


def run_optuna_for_model(preprocessor: ColumnTransformer, X_train: pd.DataFrame, y_train: pd.Series, model_name: str, n_trials: int = 30):
    """Tune hyperparameters for a single base model using Optuna optimizing PR-AUC with stratified CV.
    Returns (fitted_pipeline, best_params, best_score).
    """
    if not HAS_OPTUNA:
        raise RuntimeError("Optuna not available")
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    def objective(trial):
        if model_name == "rf":
            n_estimators = trial.suggest_int("n_estimators", 150, 600)
            max_depth = trial.suggest_int("max_depth", 3, 12)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 6)
            max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])
            cw = trial.suggest_categorical("class_weight", [None, "balanced"])
            clf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=42,
                class_weight=cw,
                n_jobs=-1,
            )
        elif model_name == "svc":
            C = trial.suggest_float("C", 1e-3, 1e3, log=True)
            gamma = trial.suggest_float("gamma", 1e-4, 1.0, log=True)
            cw = trial.suggest_categorical("class_weight", [None, "balanced"])
            clf = SVC(kernel="rbf", probability=True, C=C, gamma=gamma, class_weight=cw, random_state=42)
        elif model_name == "xgb" and HAS_XGB:
            n_estimators = trial.suggest_int("n_estimators", 200, 700)
            max_depth = trial.suggest_int("max_depth", 3, 8)
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
            subsample = trial.suggest_float("subsample", 0.5, 1.0)
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
            reg_lambda = trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True)
            clf = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                reg_lambda=reg_lambda,
                random_state=42,
                n_jobs=-1,
                eval_metric="logloss",
                use_label_encoder=False,
            )
        else:
            raise RuntimeError("Unsupported model for HPO or XGB unavailable")
        pipe = Pipeline(steps=[("preprocess", preprocessor), ("clf", clf)])
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        try:
            scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="average_precision")
            return scores.mean()
        except Exception:
            return 0.0

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = study.best_params
    # rebuild estimator with best params
    if model_name == "rf":
        clf = RandomForestClassifier(random_state=42, n_jobs=-1, **best_params)
    elif model_name == "svc":
        clf = SVC(kernel="rbf", probability=True, random_state=42, **best_params)
    else:  # xgb
        clf = XGBClassifier(random_state=42, n_jobs=-1, eval_metric="logloss", use_label_encoder=False, **best_params)
    tuned_pipe = Pipeline(steps=[("preprocess", preprocessor), ("clf", clf)])
    tuned_pipe.fit(X_train, y_train)
    return tuned_pipe, best_params, float(study.best_value)


def generate_shap_visuals(pipe: Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame, out_dir: str) -> None:
    if not HAS_SHAP or not SHAP_ENABLED:
        return
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        # Obtain preprocessor and estimator
        pre = None
        est = pipe
        if hasattr(pipe, "named_steps"):
            pre = pipe.named_steps.get("preprocess", None)
            est = pipe.named_steps.get("clf", est)
        # Transform features
        Xtr = pre.transform(X_train) if pre is not None else X_train.values
        Xte = pre.transform(X_test) if pre is not None else X_test.values
        # Subsample for speed
        n_bg = min(200, Xtr.shape[0])
        n_te = min(200, Xte.shape[0])
        bg = Xtr[:n_bg]
        Xe = Xte[:n_te]
        # Pick appropriate explainer
        is_tree = isinstance(est, (RandomForestClassifier, GradientBoostingClassifier))
        try:
            from xgboost import XGBClassifier as _XGBC  # type: ignore
            if isinstance(est, _XGBC):
                is_tree = True
        except Exception:
            pass
        if is_tree:
            explainer = shap.TreeExplainer(est)
            shap_vals = explainer.shap_values(Xe)
            # Handle multi-class outputs
            sv = shap_vals[1] if isinstance(shap_vals, list) and len(shap_vals) > 1 else (shap_vals if not isinstance(shap_vals, list) else shap_vals[0])
        else:
            f = (lambda X: est.predict_proba(X)[:, 1]) if hasattr(est, "predict_proba") else est.decision_function
            explainer = shap.KernelExplainer(f, bg)
            sv = explainer.shap_values(Xe)
        # Global summary
        plt.figure()
        shap.summary_plot(sv, Xe, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "shap_global.svg"), format="svg")
        plt.close()
        # Single-case force plot (fallback to waterfall)
        idx = 0
        try:
            fig = shap.force_plot(explainer.expected_value if not isinstance(explainer.expected_value, list) else explainer.expected_value[1], sv[idx], Xe[idx, :], matplotlib=True, show=False)
            fig.savefig(os.path.join(out_dir, "shap_case_0.svg"))
            plt.close(fig)
        except Exception:
            plt.figure()
            expected = explainer.expected_value
            if isinstance(expected, list):
                expected = expected[1] if len(expected) > 1 else expected[0]
            shap.plots._waterfall.waterfall_legacy(expected, sv[idx], show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "shap_case_0.svg"), format="svg")
            plt.close()
    except Exception:
        # fail silently to not block pipeline
        pass

    plt.close()


def main():
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "clean_data.csv")
    df = pd.read_csv(data_path)
    target_col = detect_target_column(df)
    df = df.dropna(subset=[target_col]).copy()
    df[target_col] = coerce_binary_target(df[target_col])
    df = df.dropna(subset=[target_col]).copy()

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(float)

    # New analyses for report sections 3.2 and 3.3
    try:
        analyze_missingness(df, REPORTS_DIR, VISUALS_DIR, phase_tag="phase3b")
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in X.columns if c not in num_cols]
        analyze_outliers(X, num_cols, REPORTS_DIR, VISUALS_DIR, phase_tag="phase3b")
        write_handling_strategies_report(num_cols, cat_cols, os.environ.get("PHASE3B_OUTLIER_TREATMENT", "none"), REPORTS_DIR, phase_tag="phase3b")
    except Exception as e:
        print(f"Warning: analysis steps failed: {e}")

    preprocessor = build_preprocessor(df, target_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) == 2 else None
    )

    # Train single models across augmentation samplers
    if IMBALANCE_MODE == "none":
        sampler_names = ["none"]
        cw_list = [None]
    elif IMBALANCE_MODE == "class_weight":
        sampler_names = ["none"]
        cw_list = ["balanced"]
    else:  # augmentation
        sampler_names = ["smotenc", "smote", "adasyn", "bsmote", "svmsmote", "smotetomek", "smoteenn"]
        cw_list = [None, "balanced"]
    configs = [(sampler_name, cw) for sampler_name in sampler_names for cw in cw_list]

    single_models: Dict[str, Pipeline] = {}
    metrics_rows = []
    cms: Dict[str, np.ndarray] = {}
    rocs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    pr_map: Dict[str, Tuple[np.ndarray, np.ndarray, float]] = {}

    for sampler_name, cw in configs:
        pipes = build_single_pipelines(preprocessor, sampler_name=sampler_name, cw=cw, y_train=y_train)
        for name, pipe in pipes.items():
            if "smote" in name and len(np.unique(y_train)) < 2:
                continue
            # nested-CV threshold locking
            locked_thr = _select_threshold_via_cv(pipe, X_train, y_train, n_splits=NESTED_CV_SPLITS)
            m = fit_and_score(pipe, X_train, y_train, X_test, y_test, threshold=locked_thr)
            # NEW: per-model CV F1 on training folds for generalization gap
            cv_mean, cv_std = float("nan"), float("nan")
            try:
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="f1")
                cv_mean, cv_std = float(scores.mean()), float(scores.std())
            except Exception:
                pass
            gen_gap = float(cv_mean) - float(m.get("f1", np.nan)) if not np.isnan(cv_mean) and not np.isnan(m.get("f1", np.nan)) else float("nan")
            # collect predictions for visuals
            try:
                prob = pipe.predict_proba(X_test)[:, 1]
            except Exception:
                pipe.fit(X_train, y_train)
                prob = pipe.predict_proba(X_test)[:, 1]
            pred = (prob >= locked_thr).astype(int)
            from sklearn.metrics import roc_curve as _roc_curve
            try:
                fpr, tpr, _ = _roc_curve(y_test, prob)
                rocs[name] = (fpr, tpr)
            except Exception:
                pass
            from sklearn.metrics import confusion_matrix as _confusion_matrix
            try:
                cms[name] = _confusion_matrix(y_test, pred)
            except Exception:
                pass
            single_models[name] = pipe
            row = {"model": name, **m, "cv_f1_mean": cv_mean, "cv_f1_std": cv_std, "generalization_gap": gen_gap, "thr_locked": float(locked_thr), "imbalance_mode": IMBALANCE_MODE, "calibrated": bool(CALIBRATE)}
            metrics_rows.append(row)

    # Ensembles (kept unchanged for now)
    voting_soft = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", VotingClassifier(
            estimators=[
                ("lr", LogisticRegression(max_iter=2000)),
                ("rf", RandomForestClassifier(n_estimators=300, random_state=42)),
                ("gb", GradientBoostingClassifier(random_state=42)),
            ]
            + ([ ("xgb", XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=42, n_jobs=-1, eval_metric="logloss", use_label_encoder=False)) ] if HAS_XGB else [])
            + ([ ("cat", CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1, loss_function="Logloss", random_state=42, verbose=False, allow_writing_files=False)) ] if HAS_CAT else []),
            voting="soft"
        ))
    ])

    stacking_lr = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", StackingClassifier(
            estimators=[("rf", RandomForestClassifier(n_estimators=300, random_state=42)),
                        ("gb", GradientBoostingClassifier(random_state=42))],
            final_estimator=LogisticRegression(max_iter=2000),
            stack_method="predict_proba"
        ))
    ])

    # Fit ensembles with default threshold 0.5 for visuals
    for name, pipe in {"voting_soft": voting_soft, "stacking_lr": stacking_lr}.items():
        m = fit_and_score(pipe, X_train, y_train, X_test, y_test, threshold=0.5)
        # visuals
        try:
            prob = pipe.predict_proba(X_test)[:, 1]
            pred = (prob >= 0.5).astype(int)
            fpr, tpr, _ = roc_curve(y_test, prob)
            rocs[name] = (fpr, tpr)
            prec, rec, _ = precision_recall_curve(y_test, prob)
            ap = average_precision_score(y_test, prob)
            pr_map[name] = (prec, rec, float(ap))
            cms[name] = confusion_matrix(y_test, pred)
        except Exception:
            pass
        single_models[name] = pipe
        metrics_rows.append({"model": name, **m, "thr_locked": 0.5, "imbalance_mode": IMBALANCE_MODE, "calibrated": bool(CALIBRATE)})

    # Persist metrics and JSON
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_csv = os.path.join(REPORTS_DIR, "phase3b_metrics.csv")
    metrics_json = os.path.join(REPORTS_DIR, "phase3b_metrics.json")
    metrics_df.to_csv(metrics_csv, index=False)
    with open(metrics_json, "w") as f:
        json.dump({r["model"]: {k: v for k, v in r.items() if k != "model"} for r in metrics_rows}, f, indent=2)

    # HPO for top candidates (RF, SVC, XGB) optimizing PR-AUC
    if HPO_ENABLED and HAS_OPTUNA:
        hpo_results = {}
        for mname in (["rf", "svc"] + (["xgb"] if HAS_XGB else [])):
            try:
                tuned_pipe, best_params, best_score = run_optuna_for_model(preprocessor, X_train, y_train, mname, n_trials=HPO_TRIALS)
                # Evaluate tuned model
                m = fit_and_score(tuned_pipe, X_train, y_train, X_test, y_test, threshold=0.5)
                label = f"{mname}_optuna"
                single_models[label] = tuned_pipe
                metrics_rows.append({"model": label, **m, "thr_locked": 0.5, "imbalance_mode": IMBALANCE_MODE, "calibrated": bool(CALIBRATE)})
                hpo_results[label] = {"best_params": best_params, "cv_pr_auc": best_score}
            except Exception:
                continue
        # Refresh metrics and persist HPO results
        metrics_df = pd.DataFrame(metrics_rows)
        try:
            with open(os.path.join(REPORTS_DIR, "phase3b_hpo_results.json"), "w") as f:
                json.dump(hpo_results, f, indent=2)
        except Exception:
            pass

    # NEW: visuals for confusion matrices, ROC curves, PR curves, and model comparison table
    plot_confusion_matrices(cms, os.path.join(VISUALS_DIR, "confusion_matrices.svg"))
    plot_roc_curves(rocs, os.path.join(VISUALS_DIR, "roc_curve.svg"), title="ROC Curves (Phase 3b)")
    if pr_map:
        plot_pr_curves(pr_map, os.path.join(VISUALS_DIR, "pr_curve.svg"), title="Precision-Recall Curves (Phase 3b)")
    plot_metrics_table(metrics_df, os.path.join(VISUALS_DIR, "model_comparison_table.svg"), title="Model Comparison (Phase 3b)")
    # Balanced-only comparison table
    if not metrics_df.empty:
        try:
            balanced_df = metrics_df[metrics_df["model"].astype(str).str.contains("_balanced_")]
            if not balanced_df.empty:
                plot_metrics_table(balanced_df, os.path.join(VISUALS_DIR, "model_comparison_table_balanced.svg"), title="Model Comparison (Balanced Only)")
        except Exception:
            pass

    # SHAP interpretability for best model
    if SHAP_ENABLED and HAS_SHAP and not metrics_df.empty:
        try:
            # Choose best among models we can actually access in single_models
            model_keys = set(single_models.keys())
            if model_keys:
                mdf = metrics_df[metrics_df["model"].isin(model_keys)]
                if not mdf.empty:
                    best_name = mdf.sort_values("f1", ascending=False)["model"].iloc[0]
                    best_pipe = single_models.get(best_name)
                    if best_pipe is not None:
                        generate_shap_visuals(best_pipe, X_train, X_test, VISUALS_DIR)
        except Exception:
            pass

    # Evaluate curated test cases without limiting top_k to ensure sampler coverage
    evaluate_test_cases(single_models, df, target_col, top_k=None)

    # By-augmentation summary
    if not metrics_df.empty:
        def _aug(m):
            parts = str(m).split("_")
            return parts[-1] if parts else "none"
        metrics_df["augmentation"] = metrics_df["model"].apply(_aug)
        best_by_aug = (
            metrics_df.sort_values("f1", ascending=False)
            .groupby("augmentation", as_index=False)
            .first()
        )
        best_by_aug.to_csv(os.path.join(REPORTS_DIR, "phase3b_metrics_by_augmentation.csv"), index=False)

    # Additional comparative tests: threshold sweep, 5-fold CV F1 for top models, bootstrap 95% CI on test set
    if not metrics_df.empty:
        topN = 10
        top_models = metrics_df.sort_values("f1", ascending=False)["model"].head(topN).tolist()
        thr_grid = np.linspace(0.1, 0.9, 17)
        sweep_rows: List[Dict[str, float]] = []
        cv_rows: List[Dict[str, float]] = []
        boot_rows: List[Dict[str, float]] = []

        for name in top_models:
            pipe = single_models.get(name)
            if pipe is None:
                continue
            # threshold sweep
            try:
                best_thr, best_f1, rows = threshold_sweep(pipe, X_test, y_test, thr_grid)
                for r in rows:
                    sweep_rows.append({"model": name, **r})
                sweep_rows.append({"model": name, "threshold": float("nan"), "f1": float("nan")})  # separator
                # bootstrap CI at 0.5
                mean_f1, low, high = bootstrap_f1(pipe, X_test, y_test, n_boot=200, random_state=42)
                boot_rows.append({"model": name, "mean_f1_thr0_5": mean_f1, "ci_low": low, "ci_high": high, "best_threshold": best_thr, "best_f1": best_f1})
            except Exception:
                pass
            # 5-fold CV on full data
            try:
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                scores = cross_val_score(pipe, X, y, cv=skf, scoring="f1")
                cv_rows.append({"model": name, "cv_f1_mean": float(scores.mean()), "cv_f1_std": float(scores.std())})
            except Exception:
                pass

        if sweep_rows:
            pd.DataFrame(sweep_rows).to_csv(os.path.join(REPORTS_DIR, "phase3b_threshold_sweep.csv"), index=False)
        if cv_rows:
            pd.DataFrame(cv_rows).to_csv(os.path.join(REPORTS_DIR, "phase3b_cv_f1.csv"), index=False)
        if boot_rows:
            pd.DataFrame(boot_rows).to_csv(os.path.join(REPORTS_DIR, "phase3b_bootstrap_f1_ci.csv"), index=False)

    # Generate per-test-case visuals using top models (ensembles prioritized) and persist them
    evaluate_test_cases(single_models, df, target_col, top_k=8)

    # Append/update best models summary at project root
    try:
        root_summary = os.path.join(os.path.dirname(os.path.dirname(__file__)), "best_models_summary.txt")
        with open(root_summary, "a") as f:
            f.write("\n\n=== Phase 3b Update (SMOTENC + Nested-CV Threshold Locking + Calibration) ===\n")
            f.write(f"Imbalance mode: {IMBALANCE_MODE}; Calibration: {CALIBRATE} ({CALIBRATION_METHOD})\n")
            if not metrics_df.empty:
                top3 = metrics_df.sort_values("f1", ascending=False).head(3)
                for _, r in top3.iterrows():
                    f.write(f"- {r['model']}: F1={r['f1']:.4f}, ROC_AUC={r['roc_auc']:.4f}, thr_locked={r.get('thr_locked', 0.5)}\n")
    except Exception:
        pass

    print(f"Phase 3b completed. Metrics -> {metrics_csv}")


if __name__ == "__main__":
    main()