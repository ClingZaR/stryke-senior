import os
import json
from typing import List, Dict
import datetime

from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
from io import BytesIO, StringIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

try:
    import shap  # type: ignore
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'clean_data.csv')
IST_DATA_PATH = os.path.join(BASE_DIR, 'IST_data.csv')
SUMMARY_DIR = os.path.join(BASE_DIR, 'summary')
PHASE4B_DIR = os.path.join(BASE_DIR, 'phase4b')

# Feature mapping between dashboard form fields and dataset columns
# Allow multiple candidate column names per feature to handle dataset variants
FEATURE_MAP = {
    'age': ['Age'],
    'gender': ['Gender'],
    'nationality': ['Nationality'],
    'bp_sys': ['BP_sys'],
    'bp_dia': ['BP_dia'],
    'hypertension': ['Known case of Hypertension (YES/NO)'],
    'diabetes': ['known case of diabetes (YES/NO)'],
    # Some datasets use "Blood glucose" vs "Blood glucose at admission"
    'glucose': ['Blood glucose', 'Blood glucose at admission'],
    # HbA1c can appear with or without the admission note
    'hba1c': ['HbA1c', 'HbA1c (last one before admission)'],
    # Cholesterol may be recorded as last known vs at admission
    'cholesterol': ['Total cholesterol (last one before admission)', 'Cholesterol level at admission'],
    'troponin': ['Troponin level at admission'],
    'bmi': ['BMI'],
    'atrial_fibrillation': ['known case of atrial fibrillation (YES/NO)'],
}

TARGET_COL = 'Death outcome'
THRESHOLD = 0.15  # Adjusted for more balanced precision/recall (was 0.01)

app = Flask(__name__)
CORS(app)

# Ensure CORS headers are present even on errors
@app.after_request
def add_cors_headers(resp):
    resp.headers.setdefault('Access-Control-Allow-Origin', '*')
    resp.headers.setdefault('Access-Control-Allow-Headers', 'Content-Type, Authorization')
    resp.headers.setdefault('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    return resp

# Globals for trained model and preprocessors
model: GradientBoostingClassifier = None  # type: ignore
preprocessor: ColumnTransformer = None  # type: ignore
train_cols: List[str] = []

# Globals for similar-case retrieval
cases_df: pd.DataFrame = None  # type: ignore
nn_preproc: ColumnTransformer = None  # type: ignore
nn_index: NearestNeighbors = None  # type: ignore
X_all_trans = None

# Globals for IST similar-case retrieval
ist_df: pd.DataFrame = None
ist_nn_index: NearestNeighbors = None
ist_preproc: ColumnTransformer = None
ist_X_trans = None

 
def _json_safe(v):
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if hasattr(v, 'item'):
        try:
            return v.item()
        except Exception:
            pass
    return v

def _parse_bool(val):
    """Robust boolean parsing from various input formats."""
    if val is None:
        return False
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    return str(val).lower() in ['1', 'true', 'yes', 'y', 'on']

# Map incoming form keys to actual dataset column names resolved at training time
FORM_TO_COL: Dict[str, str] = {}


# Advanced preprocessing classes from phase5b
class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    def __init__(self, min_frequency: float = 0.02):
        self.min_frequency = float(min_frequency)
        self.allowed_: Dict[object, set] = {}

    def fit(self, X, y=None):
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
        if not hasattr(X, "columns"):
            X = pd.DataFrame(X)
        X2 = X.copy()
        for c in X2.columns:
            allowed = self.allowed_.get(c, set())
            s = X2[c].astype(str)
            X2[c] = s.where(s.isin(allowed), other="Other")
        return X2


def make_ohe(min_freq: float = 0.02):
    """Safe OneHotEncoder across sklearn versions"""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse=True, min_frequency=min_freq)
    except TypeError:
        # Older sklearn without min_frequency
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def _map_to_dataset_category(val, col_name: str):
    """Return the exact dataset category string for a given value (case/whitespace-insensitive).

    If no match is found, return the original value.
    """
    global cases_df
    try:
        if cases_df is None:
            cases_df = pd.read_csv(DATA_PATH)
        series = cases_df[col_name].dropna().astype(str)
        lookup = {s.strip().lower(): s for s in series.unique()}
        return lookup.get(str(val).strip().lower(), val)
    except Exception:
        return val


def _infer_target_series(df: pd.DataFrame) -> pd.Series:
    # Detect target column robustly
    candidates = [
        'Death outcome',
        'Death outcome (YES/NO)',
        'target', 'stroke', 'death', 'outcome'
    ]
    target_col = None
    for col in candidates:
        if col in df.columns:
            target_col = col
            break
    if target_col is None:
        raise KeyError(f"Target column not found. Available columns: {list(df.columns)}")

    y = df[target_col]
    if y.dtype == 'O':
        y_lower = y.astype(str).str.lower()
        y_bin = y_lower.map({
            'yes': 1, 'true': 1, '1': 1,
            'no': 0, 'false': 0, '0': 0
        })
        y_bin = y_bin.fillna((y_lower != 'no').astype(int))
        return y_bin.astype(int)
    try:
        return (y.astype(float) > 0).astype(int)
    except Exception:
        return (pd.to_numeric(y, errors='coerce').fillna(0) > 0).astype(int)


def train_model():
    global model, preprocessor, train_cols, cases_df, FORM_TO_COL, THRESHOLD
    cases_df = pd.read_csv(DATA_PATH)
    df = cases_df

    # Resolve feature names robustly and record mapping form_key -> dataset column
    FORM_TO_COL = {}
    cols_present = []
    for form_key, desired in FEATURE_MAP.items():
        # Support both single string and list of candidates
        candidates = desired if isinstance(desired, (list, tuple)) else [desired]
        col = _find_column(df, candidates)
        if col:
            FORM_TO_COL[form_key] = col
            cols_present.append(col)
    if not cols_present:
        raise RuntimeError('Expected dashboard features not found in clean_data.csv')

    # Prepare X, y
    X = df[cols_present].copy()
    y = _infer_target_series(df)

    # Normalize categorical and numeric lists
    categorical_cols = []
    numeric_cols = []
    for col in cols_present:
        if X[col].dtype == 'O':
            categorical_cols.append(col)
        else:
            numeric_cols.append(col)

    # Advanced preprocessor with phase5b improvements
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ('rare', RareCategoryGrouper(min_frequency=0.02)),
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', make_ohe(min_freq=0.02)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols),
        ]
    )

    # Model - Using Gradient Boosting (best performing model from phase5b) with calibration
    gb = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        subsample=0.8,
    )

    # Pipeline with calibration for better probability estimates
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor), 
        ('model', CalibratedClassifierCV(gb, method='isotonic', cv=3))
    ])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf.fit(X_train, y_train)

    # Save to globals
    model = clf
    train_cols = cols_present

    # Return metrics
    y_prob = model.predict_proba(X_test)[:, 1]
    # Use the optimal threshold from phase5b analysis instead of auto-tuning
    # The phase5b analysis found 0.01 to be optimal for recall_85 and net_benefit
    # Auto-tuning here would override this carefully chosen threshold
    
    y_pred = (y_prob >= THRESHOLD).astype(int)

    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, y_prob)) if len(np.unique(y_test)) > 1 else None,
        'pr_auc': float(average_precision_score(y_test, y_prob)) if len(np.unique(y_test)) > 1 else None,
        'threshold': THRESHOLD,
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'n_test': int(len(y_test)),
    }
    return metrics


def _build_nn_index():
    """Build nearest-neighbors index over all cases using a dedicated preprocessing (with scaling)."""
    global cases_df, nn_preproc, nn_index, X_all_trans
    # Load dataset
    cases_df = pd.read_csv(DATA_PATH)
    if not train_cols:
        # Ensure model is trained to populate train_cols
        train_model()
    X_all = cases_df[train_cols].copy()

    # Derive categorical vs numeric from dtypes in X_all
    categorical_cols = [c for c in train_cols if X_all[c].dtype == 'O']
    numeric_cols = [c for c in train_cols if X_all[c].dtype != 'O']

    # Preprocessor for NN: impute + scale numeric; impute + onehot categorical
    num_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler(with_mean=False)),  # support sparse concat
    ])
    cat_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ])
    nn_preproc = ColumnTransformer(
        transformers=[
            ('num', num_pipe, numeric_cols),
            ('cat', cat_pipe, categorical_cols),
        ]
    )
    X_all_trans = nn_preproc.fit_transform(X_all)
    nn_index = NearestNeighbors(n_neighbors=5, metric='euclidean', algorithm='brute')
    nn_index.fit(X_all_trans)


def _build_ist_index():
    """Build nearest-neighbors index over IST cases."""
    global ist_df, ist_nn_index, ist_preproc, ist_X_trans
    
    if not os.path.exists(IST_DATA_PATH):
        print(f"IST data not found at {IST_DATA_PATH}")
        return

    ist_df = pd.read_csv(IST_DATA_PATH)
    
    # Features to use for similarity: AGE, SEX, RSBP (systolic), RATRIAL (afib), RCONSC (consciousness)
    # We need to handle missing values and encoding
    
    # Define columns
    numeric_cols = ['AGE', 'RSBP']
    categorical_cols = ['SEX', 'RATRIAL', 'RCONSC']
    
    # Ensure columns exist
    for col in numeric_cols + categorical_cols:
        if col not in ist_df.columns:
            # If critical columns missing, abort
            print(f"IST data missing column {col}")
            return

    X_ist = ist_df[numeric_cols + categorical_cols].copy()

    # Preprocessor
    num_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    
    cat_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ])
    
    ist_preproc = ColumnTransformer(
        transformers=[
            ('num', num_pipe, numeric_cols),
            ('cat', cat_pipe, categorical_cols),
        ]
    )
    
    ist_X_trans = ist_preproc.fit_transform(X_ist)
    ist_nn_index = NearestNeighbors(n_neighbors=1, metric='euclidean', algorithm='brute')
    ist_nn_index.fit(ist_X_trans)


def _find_similar_ist_case(payload):
    """Find most similar case in IST dataset."""
    global ist_df, ist_nn_index, ist_preproc, ist_X_trans
    
    if ist_nn_index is None:
        _build_ist_index()
        
    if ist_nn_index is None:
        return None

    try:
        # Map payload to IST features
        sample = {}
        
        # AGE
        age = payload.get('age')
        try:
            sample['AGE'] = float(age) if age else np.nan
        except ValueError:
            sample['AGE'] = np.nan
        
        # RSBP
        sbp = payload.get('bp_sys')
        try:
            sample['RSBP'] = float(sbp) if sbp else np.nan
        except ValueError:
            sample['RSBP'] = np.nan
        
        # SEX: M/F
        gender = str(payload.get('gender', '')).lower()
        if gender in ['male', 'm']:
            sample['SEX'] = 'M'
        elif gender in ['female', 'f']:
            sample['SEX'] = 'F'
        else:
            sample['SEX'] = np.nan
            
        # RATRIAL: Y/N
        afib = _parse_bool(payload.get('atrial_fib')) or _parse_bool(payload.get('atrial_fibrillation'))
        sample['RATRIAL'] = 'Y' if afib else 'N'
        
        # RCONSC: F (Fully alert), D (Drowsy), U (Unconscious)
        # We don't have this in input, assume 'F' (Fully alert) as default for similarity
        sample['RCONSC'] = 'F'
        
        # Create DataFrame
        X_sample = pd.DataFrame([sample])
        
        # Transform
        X_sample_trans = ist_preproc.transform(X_sample)
        
        # Find nearest
        dist, idx = ist_nn_index.kneighbors(X_sample_trans)
        
        # Get row
        row_idx = idx[0][0]
        row = ist_df.iloc[row_idx]
        
        # Extract details
        # Outcome: DIED (1=Dead, 0=Alive)
        died = row.get('DIED')
        outcome = 'death' if died == 1 else 'survived'
        
        # Treatments
        treatments = []
        if row.get('RXASP') == 'Y':
            treatments.append('Aspirin')
        
        hep = row.get('RXHEP')
        if hep in ['L', 'M', 'H']:
            dose_map = {'L': 'Low', 'M': 'Medium', 'H': 'High'}
            treatments.append(f"Heparin ({dose_map.get(hep, 'Unknown')} Dose)")
            
        # Patient details for display
        patient = {
            'Age': int(row['AGE']) if pd.notna(row['AGE']) else None,
            'Gender': 'Male' if row['SEX'] == 'M' else 'Female',
            'BP_sys': int(row['RSBP']) if pd.notna(row['RSBP']) else None,
            'Atrial Fibrillation': 'Yes' if row['RATRIAL'] == 'Y' else 'No',
            'Consciousness': _json_safe(row.get('RCONSC')),
            'Stroke Type': _json_safe(row.get('STYPE')),
            'Delay (hrs)': _json_safe(row.get('RDELAY'))
        }
        
        similarity = 1.0 / (1.0 + float(dist[0][0]))
        
        return {
            'dataset': 'IST Clinical Trial',
            'similarity': similarity,
            'outcome': outcome,
            'treatments_applied': treatments,
            'patient': patient,
            'distance': float(dist[0][0])
        }
        
    except Exception as e:
        print(f"Error finding IST similar case: {e}")
        return None


def _find_column(df: pd.DataFrame, candidates: List[str]) -> str:
    """Return the first matching column name (case-insensitive, trimmed), else ''."""
    cols_norm = {str(c).strip().lower(): c for c in df.columns}
    for name in candidates:
        key = str(name).strip().lower()
        if key in cols_norm:
            return cols_norm[key]
    # Partial contains match
    for name in candidates:
        key = str(name).strip().lower()
        for norm, orig in cols_norm.items():
            if key in norm:
                return orig
    return ''


@app.route('/')
def index():
    return send_from_directory('dashboard', 'index.html')


@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('dashboard', path)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


@app.route('/metrics', methods=['GET'])
def metrics():
    if model is None:
        m = train_model()
    else:
        # Recompute on existing test set
        df = pd.read_csv(DATA_PATH)
        X = df[train_cols].copy()
        y = _infer_target_series(df)
        _, X_test_local, _, y_test_local = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        y_prob = model.predict_proba(X_test_local)[:, 1]
        y_pred = (y_prob >= THRESHOLD).astype(int)
        m = {
            'accuracy': float(accuracy_score(y_test_local, y_pred)),
            'precision': float(precision_score(y_test_local, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test_local, y_pred, zero_division=0)),
            'f1': float(f1_score(y_test_local, y_pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_test_local, y_prob)) if len(np.unique(y_test_local)) > 1 else None,
            'pr_auc': float(average_precision_score(y_test_local, y_prob)) if len(np.unique(y_test_local)) > 1 else None,
            'threshold': THRESHOLD,
            'confusion_matrix': confusion_matrix(y_test_local, y_pred).tolist(),
            'n_test': int(len(y_test_local)),
        }
    return jsonify(m)


@app.route('/predict', methods=['POST'])
def predict():
    global cases_df, FORM_TO_COL
    if model is None:
        train_model()

    payload = request.get_json(force=True) or {}

    # Build a single-row DataFrame matching training columns
    sample = _extract_features(payload)

    X_sample = pd.DataFrame([sample])[train_cols]

    # Predict
    prob = float(model.predict_proba(X_sample)[0, 1])
    pred = int(prob >= THRESHOLD)

    # SHAP local explanation
    explanation = None
    if SHAP_AVAILABLE:
        try:
            # Extract trained components - handle CalibratedClassifierCV wrapper
            if hasattr(model, 'calibrated_classifiers_'):
                # CalibratedClassifierCV case - get the base estimator
                base_estimator = model.calibrated_classifiers_[0].estimator
                preproc = base_estimator.named_steps['preprocessor']
                rf = base_estimator.named_steps['model']
                print(f"Debug: rf type = {type(rf)}")
            else:
                # Direct pipeline case
                preproc = model.named_steps['preprocessor']
                rf = model.named_steps['model']
                print(f"Debug: rf type = {type(rf)}")
            
            # If rf is still a CalibratedClassifierCV, extract its base estimator
            if hasattr(rf, 'calibrated_classifiers_'):
                rf = rf.calibrated_classifiers_[0].estimator
                print(f"Debug: extracted rf type = {type(rf)}")
            
            X_trans = preproc.transform(X_sample)
            explainer = shap.TreeExplainer(rf)
            shap_values = explainer.shap_values(X_trans)
            # rf returns list for classes; we use positive class
            if isinstance(shap_values, list):
                shap_vals = shap_values[1]
            else:
                shap_vals = shap_values

            # Aggregate SHAP contributions per original feature
            contrib = {}
            # numeric feature names
            num_features = preproc.transformers_[0][2]
            cat_features = preproc.transformers_[1][2]
            # get onehot feature names
            ohe = preproc.named_transformers_['cat'].named_steps['onehot']
            cat_out_names = ohe.get_feature_names_out(cat_features)
            feature_out_names = list(num_features) + list(cat_out_names)
            for name, val in zip(feature_out_names, shap_vals[0]):
                if name in num_features:
                    base = name
                else:
                    base = next((feat for feat in cat_features if str(name).startswith(str(feat) + '_')), str(name).split('_')[0])
                contrib[base] = contrib.get(base, 0.0) + float(val)
            # Top contributions
            sorted_items = sorted(contrib.items(), key=lambda x: abs(x[1]), reverse=True)
            explanation = {
                'top_contributions': [{'feature': k, 'value': v} for k, v in sorted_items[:10]],
            }
        except Exception as e:
            print(f"SHAP Error: {e}")
            explanation = None

    result = {
        'probability': prob,
        'predicted_class': pred,
        'risk_level': 'low' if prob < 0.1 else ('medium' if prob < 0.3 else 'high'),
        'threshold': THRESHOLD,
        'explanation': explanation,
    }
    return jsonify(result)


def _recursive_json_safe(obj):
    if isinstance(obj, dict):
        return {k: _recursive_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_recursive_json_safe(v) for v in obj]
    return _json_safe(obj)

@app.route('/similar-case', methods=['POST'])
def similar_case():
    global cases_df, FORM_TO_COL
    """Return the most similar historical case based on mixed-typed nearest neighbors.

    Input: same payload as /predict
    Output: details of nearest case including outcome and treatments.
    """
    if model is None:
        train_model()
    if nn_index is None:
        _build_nn_index()

    try:
        payload = request.get_json(force=True) or {}
        
        # 1. Find similar case in Original Dataset
        original_case_result = _find_similar_original_case(payload)
        
        # 2. Find similar case in IST Dataset
        ist_case_result = _find_similar_ist_case(payload)
        
        response = {
            'original': original_case_result,
            'ist': ist_case_result
        }
        return jsonify(_recursive_json_safe(response))

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'similar_case_failed', 'message': str(e)}), 500


def _find_similar_original_case(payload):
    global cases_df, FORM_TO_COL, nn_index, nn_preproc, X_all_trans, train_cols
    
    try:
        if not train_cols:
            train_model()
            
        # Build sample row using resolved training columns
        sample = {}
        for form_key in FEATURE_MAP.keys():
            col_name = FORM_TO_COL.get(form_key, None)
            if not col_name:
                continue
            val = payload.get(form_key, None)
            if form_key == 'gender' and val is not None:
                # normalize then align to dataset's exact category spelling
                val = str(val).strip().lower()
                if val in ['m', 'male']:
                    val = 'male'
                elif val in ['f', 'female']:
                    val = 'female'
                try:
                    val = _map_to_dataset_category(val, col_name)
                except Exception:
                    pass
            if form_key == 'nationality' and val is not None:
                try:
                    val = _map_to_dataset_category(val, col_name)
                except Exception:
                    pass
            if form_key in ['hypertension', 'diabetes']:
                truthy = str(val).lower() in ['1', 'true', 'yes', 'y', 'on']
                try:
                    is_obj = cases_df[col_name].dtype == 'O'
                except Exception:
                    is_obj = False
                if is_obj:
                    try:
                        val = _map_to_dataset_category('YES' if truthy else 'NO', col_name)
                    except Exception:
                        val = 'YES' if truthy else 'NO'
                else:
                    val = 1 if truthy else 0
            if form_key == 'bmi':
                if val is None:
                    wt = payload.get('weight', None)
                    ht = payload.get('height', None)
                    try:
                        if wt is not None and ht is not None:
                            ht_m = float(ht) / 100.0
                            val = float(wt) / (ht_m ** 2)
                    except Exception:
                        val = None
            # Sanitize numeric inputs for NN as well
            try:
                is_obj = cases_df[col_name].dtype == 'O'
            except Exception:
                is_obj = False
            if not is_obj and form_key not in ['hypertension', 'diabetes']:
                if val is None or (isinstance(val, str) and val.strip() == ''):
                    val = np.nan
                else:
                    try:
                        val = float(val)
                    except Exception:
                        val = np.nan
            sample[col_name] = val

        X_sample = pd.DataFrame([sample])[train_cols]
        preproc = nn_preproc
        X_sample_trans = preproc.transform(X_sample)

        # Compute distance only over features provided (exclude missing from comparison)
        # Determine included original features
        included_features = []
        for form_key in FEATURE_MAP.keys():
            col_name = FORM_TO_COL.get(form_key, None)
            if not col_name:
                continue
            val = sample.get(col_name, None)
            try:
                is_obj = cases_df[col_name].dtype == 'O'
            except Exception:
                is_obj = False
            if is_obj:
                if val is None:
                    continue
                s = str(val).strip()
                if not s:
                    continue
                included_features.append(col_name)
            else:
                if val is None:
                    continue
                try:
                    if isinstance(val, float) and np.isnan(val):
                        continue
                except Exception:
                    pass
                included_features.append(col_name)

        # Map included features to transformed column indices
        num_features = nn_preproc.transformers_[0][2]
        cat_features = nn_preproc.transformers_[1][2]
        ohe = nn_preproc.named_transformers_['cat'].named_steps['onehot']
        cat_out_names = list(ohe.get_feature_names_out(cat_features))
        total_dims = X_all_trans.shape[1]
        mask = np.zeros(total_dims, dtype=bool)

        # Numeric: one dim per feature, ordered first
        for i, col in enumerate(num_features):
            if col in included_features:
                mask[i] = True

        # Categorical: multiple dims per feature, after numeric block
        offset = len(num_features)
        for j, name in enumerate(cat_out_names):
            base = name.split('_')[0]
            if base in included_features:
                mask[offset + j] = True

        # Convert to dense arrays for distance calculation
        try:
            sample_vec = X_sample_trans.toarray() if hasattr(X_sample_trans, 'toarray') else np.asarray(X_sample_trans)
            all_mat = X_all_trans.toarray() if hasattr(X_all_trans, 'toarray') else np.asarray(X_all_trans)
        except Exception:
            sample_vec = np.asarray(X_sample_trans)
            all_mat = np.asarray(X_all_trans)

        # Build per-feature index maps for efficient row-level masking
        num_index_map = {col: i for i, col in enumerate(num_features)}
        offset = len(num_features)
        cat_out_names = list(ohe.get_feature_names_out(cat_features))
        cat_index_map = {}
        for j, name in enumerate(cat_out_names):
            base = name.split('_')[0]
            cat_index_map.setdefault(base, []).append(offset + j)

        # Compute distances using overlap-only masking (include dims only if both sample and row have the feature)
        n_rows = all_mat.shape[0]
        best_idx = -1
        best_dist = float('inf')
        for r in range(n_rows):
            row = cases_df.iloc[r]
            # Determine overlapping features (sample provided AND row not missing)
            row_included = []
            for col in included_features:
                try:
                    is_obj = cases_df[col].dtype == 'O'
                except Exception:
                    is_obj = False
                val_row = row[col] if col in row.index else None
                if is_obj:
                    if val_row is None or (isinstance(val_row, str) and str(val_row).strip() == '') or pd.isna(val_row):
                        continue
                else:
                    if val_row is None or pd.isna(val_row):
                        continue
                row_included.append(col)

            # If nothing overlaps, treat as max distance
            if not row_included:
                continue

            # Build row-specific mask indices
            idxs = []
            for col in row_included:
                if col in num_index_map:
                    idxs.append(num_index_map[col])
                if col in cat_index_map:
                    idxs.extend(cat_index_map[col])

            if not idxs:
                continue

            # Compute masked distance for this row
            diff = all_mat[r] - sample_vec[0]
            # Zero out non-included dims
            if mask.any():
                # Start with all zeros and keep only the included indices
                keep_mask = np.zeros_like(diff, dtype=bool)
                keep_mask[np.array(idxs, dtype=int)] = True
                diff = np.where(keep_mask, diff, 0.0)
            dist_r = float(np.sqrt(np.sum(diff * diff)))
            if dist_r < best_dist:
                best_dist = dist_r
                best_idx = r

        # Select nearest index under overlap-only masked metric
        if best_idx == -1:
            # Fallback to previous global mask-based nearest
            diff = all_mat - sample_vec[0]
            if mask.any():
                diff[:, ~mask] = 0.0
            dists = np.sqrt(np.sum(diff * diff, axis=1))
            idx = int(np.argmin(dists))
            dist = float(dists[idx])
        else:
            idx = int(best_idx)
            dist = float(best_dist)

        # Extract case row
        row = cases_df.iloc[idx]

        # Outcome
        outcome_col = _find_column(cases_df, [
            'Death outcome', 'Death outcome (YES/NO)', 'death', 'outcome'
        ])
        outcome_val = None
        if outcome_col:
            cell = str(row[outcome_col]).strip().lower()
            outcome_val = 1 if cell in ['yes', 'true', '1'] else 0

        # Treatments applied: detect common treatment-related columns with YES
        treatment_candidates = [
            'IV fibrinolytic therapy is given',
            'Abciximab administered with IV tpa',
            'Aspirin administered',
            'Mechanical thrombectomy done',
        ]
        treatments = []
        for cand in treatment_candidates:
            col = _find_column(cases_df, [cand])
            if col:
                val = str(row[col]).strip().lower()
                if val in ['yes', 'true', '1']:
                    treatments.append(cand)

        # Timing info if available
        timing_cols = [
            'Fibrinolytic therapy timing',
            'Timing of CT scan',
            'Hospital admission timing',
        ]
        timings = {}
        for cand in timing_cols:
            col = _find_column(cases_df, [cand])
            if col:
                val = row[col]
                timings[cand] = str(val) if val is not None and not pd.isna(val) else None

        # Patient summary fields
        def get_field(cands: List[str]):
            col = _find_column(cases_df, cands)
            return None if not col else row[col]

        patient = {
            'CMRN': _json_safe(get_field(['CMRN'])),
            'Nationality': _json_safe(get_field(['Nationality'])),
            'Gender': _json_safe(get_field(['Gender'])),
            'Age': _json_safe(get_field(['Age'])),
            'BMI': _json_safe(get_field(['BMI'])),
            'BP_sys': _json_safe(get_field(['BP_sys'])),
            'BP_dia': _json_safe(get_field(['BP_dia'])),
            'HbA1c': _json_safe(get_field(['HbA1c', 'HbA1c (last one before admission)'])),
            'Blood glucose': _json_safe(get_field(['Blood glucose', 'Blood glucose at admission'])),
            'Troponin': _json_safe(get_field(['Troponin level at admission'])),
            'Stroke type': _json_safe(get_field(['Type of stroke (possibly system variable)'])),
        }

        # Similarity score from distance (bounded [0,1])
        similarity = 1.0 / (1.0 + dist)

        return {
            'dataset': 'Original Hospital Data',
            'index': idx,
            'distance': dist,
            'similarity': similarity,
            'outcome': 'death' if outcome_val == 1 else 'survived' if outcome_val == 0 else None,
            'treatments_applied': treatments,
            'timing': timings,
            'patient': patient,
        }
    except Exception as e:
        print(f"Error in _find_similar_original_case: {e}")
        import traceback
        traceback.print_exc()
        return None



@app.route('/feature-importance', methods=['GET'])
def feature_importance():
    if model is None:
        train_model()
    preproc = model.named_steps['preprocessor']
    rf = model.named_steps['model']

    # Feature importance from RF, aggregated per original feature
    importances = rf.feature_importances_
    num_features = preproc.transformers_[0][2]
    cat_features = preproc.transformers_[1][2]
    ohe = preproc.named_transformers_['cat'].named_steps['onehot']
    cat_out_names = ohe.get_feature_names_out(cat_features)
    feature_out_names = list(num_features) + list(cat_out_names)

    agg = {}
    for name, imp in zip(feature_out_names, importances):
        if name in num_features:
            base = name
        else:
            # map onehot back to original categorical feature prefix
            base = next((feat for feat in cat_features if str(name).startswith(str(feat) + '_')), str(name).split('_')[0])
        agg[base] = agg.get(base, 0.0) + float(imp)
    items = sorted(agg.items(), key=lambda x: x[1], reverse=True)

    return jsonify({
        'labels': [k for k, _ in items],
        'values': [v for _, v in items],
    })


@app.route('/pr-curve', methods=['GET'])
def pr_curve():
    """Return Precision-Recall curve points computed on the held-out test set."""
    if model is None:
        train_model()
    try:
        # Build test set using the same split
        df = pd.read_csv(DATA_PATH)
        X = df[train_cols].copy()
        y = _infer_target_series(df)
        _, X_test_local, _, y_test_local = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        y_prob = model.predict_proba(X_test_local)[:, 1]
        from sklearn.metrics import precision_recall_curve
        precision, recall, _ = precision_recall_curve(y_test_local, y_prob)
        points = [{'x': float(r), 'y': float(p)} for p, r in zip(precision.tolist(), recall.tolist())]
        return jsonify({
            'points': points,
            'pr_auc': float(average_precision_score(y_test_local, y_prob))
        })
    except Exception as e:
        return jsonify({'error': 'pr_curve_unavailable', 'detail': str(e)}), 500


@app.route('/calibration-curve', methods=['GET'])
def calibration_curve_endpoint():
    """Return calibration curve data (prob_true, prob_pred) for the test set."""
    if model is None:
        train_model()
    
    try:
        # Reconstruct test set (same random state as training)
        df = pd.read_csv(DATA_PATH)
        X = df[train_cols].copy()
        y = _infer_target_series(df)
        _, X_test_local, _, y_test_local = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        y_prob = model.predict_proba(X_test_local)[:, 1]
        
        prob_true, prob_pred = calibration_curve(y_test_local, y_prob, n_bins=10)
        
        return jsonify({
            'prob_true': prob_true.tolist(),
            'prob_pred': prob_pred.tolist(),
            'brier_score': float(np.mean((y_prob - y_test_local) ** 2))
        })
    except Exception as e:
        return jsonify({'error': 'calibration_curve_failed', 'detail': str(e)}), 500


@app.route('/shap-global', methods=['GET'])
def shap_global():
    """Return global feature importance using mean(|SHAP|) aggregated per original feature."""
    if model is None:
        train_model()
    if not SHAP_AVAILABLE:
        return jsonify({'error': 'shap_unavailable'}), 501
    try:
        # Extract trained components - handle CalibratedClassifierCV wrapper
        if hasattr(model, 'calibrated_classifiers_'):
            # CalibratedClassifierCV case - get the base estimator
            base_estimator = model.calibrated_classifiers_[0].estimator
            preproc = base_estimator.named_steps['preprocessor']
            rf = base_estimator.named_steps['model']
        else:
            # Direct pipeline case
            preproc = model.named_steps['preprocessor']
            rf = model.named_steps['model']

        # If rf is still a CalibratedClassifierCV, extract its base estimator
        if hasattr(rf, 'calibrated_classifiers_'):
            rf = rf.calibrated_classifiers_[0].estimator

        # Use test split for global explanation
        df = pd.read_csv(DATA_PATH)
        X = df[train_cols].copy()
        y = _infer_target_series(df)
        _, X_test_local, _, _ = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_trans = preproc.transform(X_test_local)
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_trans)
        if isinstance(shap_values, list):
            shap_vals = shap_values[1]
        else:
            shap_vals = shap_values

        # Names for transformed features
        num_features = preproc.transformers_[0][2]
        cat_features = preproc.transformers_[1][2]
        ohe = preproc.named_transformers_['cat'].named_steps['onehot']
        cat_out_names = ohe.get_feature_names_out(cat_features)
        feature_out_names = list(num_features) + list(cat_out_names)

        # Mean absolute SHAP per transformed feature
        mean_abs = np.mean(np.abs(np.asarray(shap_vals)), axis=0)

        # Aggregate back to original feature bases
        agg = {}
        for name, val in zip(feature_out_names, mean_abs):
            if name in num_features:
                base = name
            else:
                base = next((feat for feat in cat_features if str(name).startswith(str(feat) + '_')), str(name).split('_')[0])
            agg[base] = agg.get(base, 0.0) + float(val)
        items = sorted(agg.items(), key=lambda x: x[1], reverse=True)

        return jsonify({
            'labels': [k for k, _ in items],
            'values': [v for _, v in items]
        })
    except Exception as e:
        return jsonify({'error': 'shap_global_unavailable', 'detail': str(e)}), 500


@app.route('/treatment-recommendations', methods=['POST'])
def treatment_recommendations():
    """Generate ML-based treatment recommendations using Phase 6b model."""
    try:
        payload = request.get_json()
        
        # Load Phase 6b model trained on IST database
        import joblib
        phase6b_model_path = os.path.join(BASE_DIR, 'phase6b', 'models', 'treatment_model_IST.pkl')
        phase6b_features_path = os.path.join(BASE_DIR, 'phase6b', 'models', 'feature_names_IST.pkl')
        phase6b_cols_path = os.path.join(BASE_DIR, 'phase6b', 'models', 'feature_cols_IST.pkl')
        
        if not os.path.exists(phase6b_model_path):
            # Fallback to rule-based if model not trained
            return jsonify({'error': 'phase6b_model_not_found', 'message': 'Phase 6b IST model not trained yet'}), 503
        
        treatment_model = joblib.load(phase6b_model_path)
        treatment_feature_names = joblib.load(phase6b_features_path)
        treatment_feature_cols = joblib.load(phase6b_cols_path)
        
        # Prepare patient features for IST model
        # IST model uses: age_clean, sex_binary, sbp_clean, atrial_fib, consciousness,
        # aspirin_given, heparin_given, heparin_dose, stroke types, delay_hours
        patient_data = {}
        
        # Map form data to IST model features
        age = float(payload.get('age', 0)) if payload.get('age') else 65
        patient_data['age_clean'] = age
        
        gender = payload.get('gender', '').lower()
        patient_data['sex_binary'] = 1 if gender in ['male', 'm'] else 0
        
        bp_sys = float(payload.get('bp_sys', 0)) if payload.get('bp_sys') else 140
        patient_data['sbp_clean'] = bp_sys
        
        atrial_fib = _parse_bool(payload.get('atrial_fib')) or _parse_bool(payload.get('atrial_fibrillation'))
        patient_data['atrial_fib'] = 1 if atrial_fib else 0
        
        # Consciousness - default to 0 (fully alert)
        patient_data['consciousness'] = 0
        
        # Treatment timing
        patient_data['delay_hours'] = float(payload.get('treatment_timing', 6)) if payload.get('treatment_timing') else 6
        
        # Aspirin
        patient_data['aspirin_given'] = 1 if _parse_bool(payload.get('aspirin_administered')) else 0
        
        # Heparin
        heparin_bool = _parse_bool(payload.get('heparin_administered'))
        patient_data['heparin_given'] = 1 if heparin_bool else 0
        patient_data['heparin_dose'] = 1 if heparin_bool else 0  # Low dose default
        
        # Stroke types - default to PACS (most common)
        patient_data['stroke_PACS'] = 1
        patient_data['stroke_LACS'] = 0
        patient_data['stroke_POCS'] = 0
        patient_data['stroke_OTH'] = 0
        patient_data['stroke_TACS'] = 0
        
        # Ensure all expected columns are present (robustness against extra stroke types etc)
        for col in treatment_feature_cols:
            if col not in patient_data:
                patient_data[col] = 0
        
        # Create DataFrame with correct feature order
        X_input = pd.DataFrame([patient_data])[treatment_feature_cols]
        
        # Get predictions for different treatment scenarios
        recommendations = {}
        
        # Scenario 1: With Aspirin
        X_with_aspirin = X_input.copy()
        asp_idx = treatment_feature_cols.index('aspirin_given') if 'aspirin_given' in treatment_feature_cols else None
        if asp_idx is not None:
            X_with_aspirin.iloc[:, asp_idx] = 1
        prob_with_aspirin = treatment_model.predict_proba(X_with_aspirin)[0, 1]
        
        # Scenario 2: Without Aspirin  
        X_without_aspirin = X_input.copy()
        if asp_idx is not None:
            X_without_aspirin.iloc[:, asp_idx] = 0
        prob_without_aspirin = treatment_model.predict_proba(X_without_aspirin)[0, 1]
        
        # Calculate benefit
        aspirin_benefit = prob_without_aspirin - prob_with_aspirin
        aspirin_relative_reduction = (aspirin_benefit / prob_without_aspirin * 100) if prob_without_aspirin > 0 else 0
        
        # Determine recommendations based on ML predictions from IST trial data
        recommendations['aspirin'] = {
            'recommended': bool(aspirin_benefit > 0.01 and bp_sys < 180),  # ML shows >1% benefit and safe BP
            'confidence': float(min(max((aspirin_benefit * 30), 0), 0.95)),  # Scale benefit to confidence (adjusted x30)
            'reasoning': [
                f'IST-trained model predicts {abs(aspirin_relative_reduction):.1f}% {"reduction" if aspirin_benefit > 0 else "increase"} in mortality risk',
                f'Risk with aspirin: {prob_with_aspirin*100:.1f}%',
                f'Risk without aspirin: {prob_without_aspirin*100:.1f}%',
                f'Trained on 19,435 patients from IST clinical trial'
            ]
        }
        
        if bp_sys > 180:
            recommendations['aspirin']['recommended'] = False
            recommendations['aspirin']['reasoning'].append('⚠️ High BP (>180) increases bleeding risk - contraindication')
        elif aspirin_benefit > 0.02:
            recommendations['aspirin']['reasoning'].append('✓ Significant mortality benefit predicted')
        
        # Heparin recommendations based on IST model (model learned heparin less effective)
        hep_idx = treatment_feature_cols.index('heparin_given') if 'heparin_given' in treatment_feature_cols else None
        if hep_idx is not None:
            X_with_hep = X_input.copy()
            X_with_hep.iloc[:, hep_idx] = 1
            prob_with_hep = treatment_model.predict_proba(X_with_hep)[0, 1]
            
            X_without_hep = X_input.copy()
            X_without_hep.iloc[:, hep_idx] = 0
            prob_without_hep = treatment_model.predict_proba(X_without_hep)[0, 1]
            
            heparin_benefit = prob_without_hep - prob_with_hep
            
            recommendations['heparin'] = {
                'recommended': bool(heparin_benefit > 0.015 and atrial_fib and bp_sys < 160),
                'confidence': float(min(max((heparin_benefit * 30), 0), 0.7)), # Adjusted x30
                'reasoning': [
                    f'IST model predicts {abs(heparin_benefit*100):.1f}% {"reduction" if heparin_benefit > 0 else "increase"} in mortality',
                    f'Risk with heparin: {prob_with_hep*100:.1f}%',
                    f'Risk without heparin: {prob_without_hep*100:.1f}%',
                ]
            }
        else:
            # Conservative heparin recommendation
            recommendations['heparin'] = {
                'recommended': bool(atrial_fib and age > 70 and bp_sys < 160),
                'confidence': float(0.5 if (atrial_fib and age > 70) else 0.2),
                'reasoning': ['Based on atrial fibrillation indication', f'Age: {age} years']
            }
        
        # Combination therapy
        if recommendations['aspirin']['recommended'] and recommendations['heparin']['recommended']:
            recommendations['combination'] = {
                'recommended': True,
                'confidence': float(min(recommendations['aspirin']['confidence'], recommendations['heparin']['confidence']) * 0.9),
                'reasoning': ['Both treatments show benefit', 'Synergistic effect expected']
            }
        else:
            recommendations['combination'] = {
                'recommended': False,
                'confidence': float(0.2),
                'reasoning': ['Single agent therapy preferred based on risk profile']
            }
        
        return jsonify(recommendations)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'treatment_recommendations_failed', 'message': str(e)}), 500


@app.route('/phase-comparison', methods=['GET'])
def phase_comparison():
    # All phases with documented metrics
    phases = ['Phase 1', 'Phase 2b', 'Phase 3b', 'Phase 4b', 'Phase 5b', 'Phase 6b']
    f1s = [None, 0.421, 0.593, 0.487, None, None]  # Phase 1, 5b, 6b don't optimize F1
    aucs = [None, 0.759, 0.810, 0.780, 0.775, 0.788]
    
    # Try to load actual data from files if available
    try:
        csv_path = os.path.join(SUMMARY_DIR, 'phase2b_vs_phase3b_metrics.csv')
        df = pd.read_csv(csv_path)
        for phase_name in df['phase'].unique():
            sub = df[df['phase'] == phase_name]
            phase_str = str(phase_name)
            if 'phase2b' in phase_str.lower() or '2b' in phase_str:
                idx = 1
            elif 'phase3b' in phase_str.lower() or '3b' in phase_str:
                idx = 2
            else:
                continue
            f1s[idx] = float(sub['f1'].mean())
            aucs[idx] = float(sub['roc_auc'].mean())
    except Exception:
        pass

    # Phase 4b
    try:
        csv_path = os.path.join(PHASE4B_DIR, 'reports', 'phase4b_sampler_comparison.csv')
        df4 = pd.read_csv(csv_path)
        best = df4.sort_values('test_f1', ascending=False).iloc[0]
        f1s[3] = float(best['test_f1'])
        aucs[3] = float(best.get('test_roc_auc', best.get('roc_auc', 0.780)))
    except Exception:
        pass
    
    # Phase 5b
    try:
        json_path = os.path.join('phase5b', 'reports', 'monotone_lgb_results.json')
        with open(json_path, 'r') as f:
            p5_data = json.load(f)
            aucs[4] = float(p5_data['cv_metrics']['roc_auc_cv'])
    except Exception:
        pass
    
    # Phase 6b IST model
    try:
        metadata_path = os.path.join('phase6b', 'models', 'metadata_IST.pkl')
        metadata = joblib.load(metadata_path)
        aucs[5] = float(metadata.get('test_auc', 0.788))
    except Exception:
        pass

    return jsonify({
        'phases': phases,
        'f1': f1s,
        'roc_auc': aucs,
        'descriptions': [
            'Data Exploration',
            'Baseline + Leak-Free Pipeline',
            'SMOTE + Threshold Optimization',
            'Hyperparameter Optimization',
            'Clinical Calibration + Monotonic Constraints',
            'Treatment Effect Estimation (IST)'
        ]
    })


# ReportLab PDF generation example
@app.route('/generate-pdf', methods=['POST'])
def generate_pdf():
    try:
        # Sample data - in practice, use data from request or database
        patient_data = {
            'name': 'John Doe',
            'age': 65,
            'gender': 'Male',
            'nationality': 'American',
            'bmi': 28.4,
            'hypertension': 'Yes',
            'diabetes': 'No',
            'cholesterol': 240,
            'troponin': 0.05,
            'hba1c': 7.2,
            'glucose': 180,
            'bp_sys': 150,
            'bp_dia': 90,
            'atrial_fibrillation': 'No',
            'treatments': ['Aspirin', 'Statins'],
            'outcome': 'Survived',
            'similarity_score': 0.87,
            'recommendations': {
                'aspirin': 'Continue',
                'statins': 'Consider high-intensity statin',
                'lifestyle': 'Diet and exercise',
            }
        }
        
        # Create PDF document
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        Story = []

        # Title
        title = Paragraph("Patient Report", styles['Title'])
        Story.append(title)
        Story.append(Spacer(1, 0.2 * inch))

        # Patient information table
        patient_info = [
            ['Patient Name', patient_data['name']],
            ['Age', patient_data['age']],
            ['Gender', patient_data['gender']],
            ['Nationality', patient_data['nationality']],
            ['BMI', patient_data['bmi']],
            ['Hypertension', patient_data['hypertension']],
            ['Diabetes', patient_data['diabetes']],
            ['Cholesterol', patient_data['cholesterol']],
            ['Troponin', patient_data['troponin']],
            ['HbA1c', patient_data['hba1c']],
            ['Blood glucose', patient_data['glucose']],
            ['Blood Pressure (Sys)', patient_data['bp_sys']],
            ['Blood Pressure (Dia)', patient_data['bp_dia']],
            ['Atrial Fibrillation', patient_data['atrial_fibrillation']],
        ]
        t = Table(patient_info)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        Story.append(t)
        Story.append(Spacer(1, 0.2 * inch))

        # Outcome and recommendations
        outcome = Paragraph(f"Outcome: {patient_data['outcome']}", styles['Heading2'])
        Story.append(outcome)
        Story.append(Spacer(1, 0.1 * inch))

        recommendations_title = Paragraph("Treatment Recommendations", styles['Heading2'])
        Story.append(recommendations_title)
        for key, value in patient_data['recommendations'].items():
            rec = Paragraph(f"- {key.capitalize()}: {value}", styles['BodyText'])
            Story.append(rec)
        Story.append(Spacer(1, 0.2 * inch))

        # Similarity score
        similarity = Paragraph(f"Similarity Score: {patient_data['similarity_score']:.2f}", styles['Heading2'])
        Story.append(similarity)

        # Build PDF
        doc.build(Story)
        buffer.seek(0)

        # Send PDF file
        return send_file(buffer, as_attachment=True, download_name="patient_report.pdf", mimetype="application/pdf")
    
    except Exception as e:
        return jsonify({'error': 'PDF generation failed', 'message': str(e)}), 500


@app.route('/generate-report', methods=['POST'])
def generate_report():
    """Generate a PDF report for the patient."""
    try:
        data = request.get_json()
        patient = data.get('patient', {})
        prediction = data.get('prediction', {})
        treatments = data.get('treatments', {})
        
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
        
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name='CenterTitle', parent=styles['Heading1'], alignment=1, spaceAfter=20))
        styles.add(ParagraphStyle(name='SectionHeader', parent=styles['Heading2'], spaceBefore=15, spaceAfter=10, textColor=colors.HexColor('#1976D2')))
        styles.add(ParagraphStyle(name='RiskHigh', parent=styles['Normal'], textColor=colors.red, fontName='Helvetica-Bold'))
        styles.add(ParagraphStyle(name='RiskMedium', parent=styles['Normal'], textColor=colors.orange, fontName='Helvetica-Bold'))
        styles.add(ParagraphStyle(name='RiskLow', parent=styles['Normal'], textColor=colors.green, fontName='Helvetica-Bold'))
        
        story = []
        
        # Header
        story.append(Paragraph("STRYKE-AI Patient Report", styles['CenterTitle']))
        story.append(Paragraph(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Patient Demographics
        story.append(Paragraph("Patient Demographics", styles['SectionHeader']))
        demo_data = [
            ['Age', str(patient.get('age', 'N/A')), 'Gender', str(patient.get('gender', 'N/A')).title()],
            ['Nationality', str(patient.get('nationality', 'N/A')).title(), 'BMI', f"{float(patient.get('bmi', 0)):.1f}" if patient.get('bmi') else 'N/A'],
            ['Systolic BP', str(patient.get('bp_sys', 'N/A')), 'Diastolic BP', str(patient.get('bp_dia', 'N/A'))],
        ]
        t = Table(demo_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        t.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('BACKGROUND', (0,0), (0,-1), colors.whitesmoke),
            ('BACKGROUND', (2,0), (2,-1), colors.whitesmoke),
            ('PADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(t)
        
        # Risk Assessment
        story.append(Paragraph("Mortality Risk Assessment", styles['SectionHeader']))
        prob = float(prediction.get('probability', 0))
        risk_level = prediction.get('risk_level', 'low')
        
        risk_style = styles['RiskLow']
        if risk_level == 'high': risk_style = styles['RiskHigh']
        elif risk_level == 'medium': risk_style = styles['RiskMedium']
        
        story.append(Paragraph(f"Calculated Mortality Risk: <b>{prob*100:.1f}%</b>", styles['Normal']))
        story.append(Paragraph(f"Risk Level: {risk_level.upper()}", risk_style))
        
        if prediction.get('ci'):
            ci = prediction['ci']
            story.append(Paragraph(f"95% Confidence Interval: {ci[0]*100:.1f}% - {ci[1]*100:.1f}%", styles['Normal']))
            
        story.append(Spacer(1, 12))
        
        # Key Risk Factors (SHAP)
        if prediction.get('contributions'):
            story.append(Paragraph("Key Risk Factors (Top Contributors)", styles['SectionHeader']))
            shap_data = [['Feature', 'Impact']]
            for item in prediction['contributions'][:5]:
                val = float(item.get('value', 0))
                impact = "Increases Risk" if val > 0 else "Decreases Risk"
                shap_data.append([item.get('feature', ''), impact])
            
            t_shap = Table(shap_data, colWidths=[4*inch, 2*inch])
            t_shap.setStyle(TableStyle([
                ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
                ('PADDING', (0,0), (-1,-1), 6),
            ]))
            story.append(t_shap)
            
        # Treatment Recommendations
        story.append(Paragraph("Clinical Recommendations (Phase 6b)", styles['SectionHeader']))
        
        rec_data = [['Treatment', 'Recommendation', 'Confidence']]
        
        def add_rec_row(name, data):
            rec = "Recommended" if data.get('recommended') else "Not Recommended"
            conf = f"{float(data.get('confidence', 0))*100:.0f}%"
            return [name, rec, conf]
            
        if treatments.get('aspirin'):
            rec_data.append(add_rec_row('Aspirin', treatments['aspirin']))
        if treatments.get('heparin'):
            rec_data.append(add_rec_row('Heparin', treatments['heparin']))
        if treatments.get('combination'):
            rec_data.append(add_rec_row('Combination', treatments['combination']))
            
        t_rec = Table(rec_data, colWidths=[2*inch, 2.5*inch, 1.5*inch])
        t_rec.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('TEXTCOLOR', (1,1), (1,-1), colors.black),
        ]))
        
        # Color code recommendations
        for i, row in enumerate(rec_data[1:], 1):
            if row[1] == "Recommended":
                t_rec.setStyle(TableStyle([('TEXTCOLOR', (1,i), (1,i), colors.green)]))
            else:
                t_rec.setStyle(TableStyle([('TEXTCOLOR', (1,i), (1,i), colors.red)]))
                
        story.append(t_rec)
        
        # Disclaimer
        story.append(Spacer(1, 30))
        story.append(Paragraph("DISCLAIMER: This report is generated by an AI support tool (STRYKE-AI). It is intended for informational purposes only and does not constitute a medical diagnosis or treatment plan. All clinical decisions should be made by qualified healthcare professionals.", styles['Italic']))
        
        doc.build(story)
        buffer.seek(0)
        
        return send_file(
            buffer,
            as_attachment=True,
            download_name=f"STRYKE_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mimetype='application/pdf'
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'report_generation_failed', 'detail': str(e)}), 500


def _extract_features(payload):
    """Extract and clean features from a payload dict (single row)."""
    global cases_df, FORM_TO_COL
    # Ensure dataset is loaded for dtype checks
    if cases_df is None:
        try:
            cases_df = pd.read_csv(DATA_PATH)
        except Exception:
            pass
            
    sample = {}
    for form_key in FEATURE_MAP.keys():
        col_name = FORM_TO_COL.get(form_key, None)
        if not col_name:
            continue
            
        # Try multiple keys: form_key, col_name, lowercase versions
        val = payload.get(form_key)
        if val is None:
            val = payload.get(col_name)
        if val is None:
            # Try case-insensitive match for CSV headers
            for k in payload.keys():
                if k.lower() == form_key.lower() or k.lower() == col_name.lower():
                    val = payload[k]
                    break
                    
        if form_key == 'gender' and val is not None:
            # normalize gender strings, then align to dataset category form
            val = str(val).strip().lower()
            if val in ['m', 'male']:
                val = 'male'
            elif val in ['f', 'female']:
                val = 'female'
            try:
                val = _map_to_dataset_category(val, col_name)
            except Exception:
                pass
        if form_key == 'nationality' and val is not None:
            # align to dataset category representation for consistent one-hot
            try:
                val = _map_to_dataset_category(val, col_name)
            except Exception:
                pass
        if form_key in ['hypertension', 'diabetes']:
            # Map to training dtype: YES/NO strings if categorical; else 1/0
            truthy = str(val).lower() in ['1', 'true', 'yes', 'y', 'on']
            try:
                is_obj = cases_df[col_name].dtype == 'O'
            except Exception:
                is_obj = False
            if is_obj:
                # Align to exact dataset category representation to avoid one-hot mismatch
                try:
                    val = _map_to_dataset_category('YES' if truthy else 'NO', col_name)
                except Exception:
                    val = 'YES' if truthy else 'NO'
            else:
                val = 1 if truthy else 0
        if form_key == 'bmi':
            # if not provided, compute from weight/height if available
            if val is None:
                wt = payload.get('weight', None)
                ht = payload.get('height', None)
                try:
                    if wt is not None and ht is not None:
                        ht_m = float(ht) / 100.0
                        val = float(wt) / (ht_m ** 2)
                except Exception:
                    val = None
        # Sanitize numeric inputs: convert blanks to NaN and coerce to float
        try:
            is_obj = cases_df[col_name].dtype == 'O'
        except Exception:
            is_obj = False
        if not is_obj and form_key not in ['hypertension', 'diabetes']:
            if val is None or (isinstance(val, str) and val.strip() == ''):
                val = np.nan
            else:
                try:
                    val = float(val)
                except Exception:
                    val = np.nan
        sample[col_name] = val
    return sample


@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """Handle CSV upload for batch prediction."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if model is None:
        train_model()
        
    try:
        # Read CSV
        # Try reading with different encodings if default fails
        try:
            df = pd.read_csv(file)
        except UnicodeDecodeError:
            file.seek(0)
            df = pd.read_csv(file, encoding='latin1')
            
        results = []
        risk_counts = {'low': 0, 'medium': 0, 'high': 0}
        
        for index, row in df.iterrows():
            # Convert row to dict, handling NaNs
            row_dict = row.to_dict()
            # Clean NaNs
            row_dict = {k: (v if pd.notna(v) else None) for k, v in row_dict.items()}
            
            # Extract features using the helper
            sample = _extract_features(row_dict)
            
            # Predict
            X_sample = pd.DataFrame([sample])[train_cols]
            prob = float(model.predict_proba(X_sample)[0, 1])
            
            risk_level = 'low' if prob < 0.1 else ('medium' if prob < 0.3 else 'high')
            
            # Append result to row_dict (preserving original data)
            row_dict['Mortality_Probability'] = round(prob, 4)
            row_dict['Risk_Level'] = risk_level
            results.append(row_dict)
            risk_counts[risk_level] += 1
            
        # Create result DataFrame
        result_df = pd.DataFrame(results)
        
        # Convert to CSV string
        output = StringIO()
        result_df.to_csv(output, index=False)
        csv_content = output.getvalue()
        
        return jsonify({
            'summary': risk_counts,
            'csv_data': csv_content,
            'total_patients': len(results)
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Train at startup
    try:
        startup_metrics = train_model()
        print('Model trained. Startup metrics:', startup_metrics)
    except Exception as e:
        print('Failed to train model at startup:', e)
    # Run without the Werkzeug reloader to make it easier to run programmatically
    # and avoid duplicate processes when invoked from automated tests.
    app.run(host='127.0.0.1', port=5000, debug=False)