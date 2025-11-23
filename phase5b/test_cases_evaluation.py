import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import importlib.util
import sys

# Local imports
try:
    from .utils import prepare_dataset
except ImportError:
    from utils import prepare_dataset

try:
    from catboost import CatBoostClassifier, Pool  # type: ignore
    HAS_CAT = True
except Exception:
    HAS_CAT = False


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
PHASE_DIR = os.path.join(PROJECT_ROOT, "phase5b")
REPORTS_DIR = os.path.join(PHASE_DIR, "reports")
VISUALS_DIR = os.path.join(PHASE_DIR, "visuals")
MODELS_DIR = os.path.join(PHASE_DIR, "models")
DATA_PATH = os.path.join(PROJECT_ROOT, "clean_data.csv")

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(VISUALS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def _ensure_custom_classes_for_unpickle():
    """Ensure custom transformers used in pickled pipelines are available on __main__.
    This avoids AttributeError when joblib loads pipelines saved with __main__.RareCategoryGrouper.
    """
    try:
        pipeline_path = os.path.join(PHASE_DIR, 'pipeline.py')
        spec = importlib.util.spec_from_file_location('phase5b_pipeline', pipeline_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore
        main_mod = sys.modules.get('__main__') or sys.modules.get(__name__)
        for cls_name in ['RareCategoryGrouper']:
            if hasattr(mod, cls_name) and main_mod is not None:
                setattr(main_mod, cls_name, getattr(mod, cls_name))
    except Exception:
        pass


def detect_target_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        name = str(col).strip().lower()
        if "death" in name and "outcome" in name:
            return col
    for col in df.columns:
        name = str(col).strip().lower()
        if "death" in name and all(k not in name for k in ["timing", "therapy", "level", "type"]):
            return col
    candidates = ["Death outcome (YES/NO)", "target", "label", "y", "stroke", "death", "outcome"]
    for c in candidates:
        if c in df.columns:
            return c
    return df.columns[-1]


def describe_case(r: pd.Series) -> str:
    parts = []
    sex = r.get("Gender")
    age = r.get("Age")
    if isinstance(sex, str) and sex:
        parts.append(sex)
    if pd.notna(age):
        try:
            parts.append(str(int(age)))
        except Exception:
            parts.append(str(age))

    conds = []
    yes_flags = [
        ("Known case of Hypertension (YES/NO)", "HTN"),
        ("known case of diabetes (YES/NO)", "DM"),
        ("known case of atrial fibrillation (YES/NO)", "AF"),
        ("known case of coronary heart disease (YES/NO)", "CHD"),
        ("Are they on warfarin, or heparin before admission (YES/NO)", "warfarin/heparin"),
        ("Personal previous history of stoke (YES/NO)", "prior stroke"),
        ("Personal previous history of Transient Ischemic Attack (YES/NO)", "prior TIA"),
        ("oral contraceptive use in female(YES/NO)", "OCP"),
        ("IV fibrinolytic therapy is given (YES/NO)", "IV tPA"),
        ("Mechanical thrombectomy done (YES/NO)", "thrombectomy"),
    ]
    for col, label in yes_flags:
        val = r.get(col)
        if isinstance(val, str) and val.strip().upper() == "YES":
            conds.append(label)

    sys = r.get("BP_sys"); dia = r.get("BP_dia")
    if pd.notna(sys) and pd.notna(dia):
        try:
            s, d = int(sys), int(dia)
        except Exception:
            s, d = sys, dia
        if (isinstance(s, (int, float)) and s >= 160) or (isinstance(d, (int, float)) and d >= 100):
            conds.append(f"BP {s}/{d}")
        elif (isinstance(s, (int, float)) and s <= 95) or (isinstance(d, (int, float)) and d <= 60):
            conds.append(f"BP {s}/{d}")

    tro = r.get("Troponin level at admission")
    if pd.notna(tro) and isinstance(tro, (int, float)) and tro >= 0.10:
        conds.append(f"Troponin {tro:.2f}")

    glu = r.get("Blood glucose at admission")
    if pd.notna(glu) and isinstance(glu, (int, float)) and glu >= 200:
        try:
            conds.append(f"Glucose {int(glu)}")
        except Exception:
            conds.append(f"Glucose {glu}")

    hba1c = r.get("HbA1c (last one before admission)")
    if pd.notna(hba1c) and isinstance(hba1c, (int, float)) and hba1c >= 6.5:
        conds.append(f"HbA1c {hba1c:.1f}")

    inr = r.get("INR")
    if pd.notna(inr) and isinstance(inr, (int, float)) and inr >= 2.0:
        conds.append(f"INR {inr:.1f}")

    chol_tot = r.get("Total cholesterol (last one before admission)")
    chol_adm = r.get("Cholesterol level at admission")
    chol_val = chol_adm if pd.notna(chol_adm) else chol_tot
    if pd.notna(chol_val) and isinstance(chol_val, (int, float)) and chol_val >= 240:
        try:
            conds.append(f"Chol {int(chol_val)}")
        except Exception:
            conds.append(f"Chol {chol_val}")

    bmi = r.get("BMI")
    if pd.notna(bmi) and isinstance(bmi, (int, float)):
        if bmi >= 30:
            conds.append("obese")
        elif bmi < 18.5:
            conds.append("underweight")

    ct_timing = r.get("Timing of CT scan")
    if isinstance(ct_timing, str) and ct_timing.strip():
        conds.append(f"CT {ct_timing}")
    fibr_timing = r.get("Fibrinolytic therapy timing")
    if isinstance(fibr_timing, str) and fibr_timing.strip():
        conds.append(f"tPA {fibr_timing}")

    head = " ".join(parts) if parts else "Case"
    desc = ", ".join(conds) if conds else "no comorbidities"
    label = f"{head} — {desc}"
    return label[:120]


def build_test_cases(all_cols: list[str]) -> pd.DataFrame:
    base = {c: np.nan for c in all_cols}

    # Ensure BMI can be derived; include height/weight defaults where available
    for k in ["Height", "Weight on admission"]:
        if k in base:
            base[k] = 1.70 if k == "Height" else 70.0

    def row(update: dict):
        r = base.copy()
        r.update(update)
        return r

    cases = [
        # 1) Low-risk young female, normal vitals
        row({
            "Gender": "Female", "Age": 25, "BMI": 22.0, "BP_sys": 110, "BP_dia": 70,
            "Known case of Hypertension (YES/NO)": "NO", "known case of diabetes (YES/NO)": "NO",
            "known case of atrial fibrillation (YES/NO)": "NO", "Troponin level at admission": 0.01,
            "Blood glucose at admission": 95, "HbA1c (last one before admission)": 5.2,
            "INR": 1.0
        }),
        # 2) Middle-aged male with hypertension
        row({
            "Gender": "Male", "Age": 50, "BMI": 27.5, "BP_sys": 145, "BP_dia": 90,
            "Known case of Hypertension (YES/NO)": "YES", "known case of diabetes (YES/NO)": "NO",
            "known case of atrial fibrillation (YES/NO)": "NO", "Blood glucose at admission": 105,
            "HbA1c (last one before admission)": 5.8, "INR": 1.1
        }),
        # 3) Older male, AFib + diabetes, high glucose/HbA1c
        row({
            "Gender": "Male", "Age": 72, "BMI": 29.0, "BP_sys": 150, "BP_dia": 92,
            "known case of atrial fibrillation (YES/NO)": "YES", "Known case of Hypertension (YES/NO)": "YES",
            "known case of diabetes (YES/NO)": "YES", "Blood glucose at admission": 210,
            "HbA1c (last one before admission)": 8.9, "Troponin level at admission": 0.08,
            "INR": 1.2
        }),
        # 4) Older female, OCP yes, moderate BP
        row({
            "Gender": "Female", "Age": 68, "BMI": 24.0, "BP_sys": 135, "BP_dia": 85,
            "oral contraceptive use in female(YES/NO)": "YES", "Known case of Hypertension (YES/NO)": "YES",
            "known case of diabetes (YES/NO)": "NO", "Blood glucose at admission": 115,
            "HbA1c (last one before admission)": 6.0
        }),
        # 5) Obese, high cholesterol
        row({
            "Gender": "Male", "Age": 60, "BMI": 34.0, "BP_sys": 140, "BP_dia": 88,
            "Total cholesterol (last one before admission)": 260,
            "Cholesterol level at admission": 240,
            "Known case of Hypertension (YES/NO)": "YES",
            "known case of diabetes (YES/NO)": "NO"
        }),
        # 6) Underweight, low BP
        row({
            "Gender": "Female", "Age": 45, "BMI": 17.5, "BP_sys": 95, "BP_dia": 60,
            "Known case of Hypertension (YES/NO)": "NO", "known case of diabetes (YES/NO)": "NO"
        }),
        # 7) High INR (bleeding risk), normal other labs
        row({
            "Gender": "Male", "Age": 55, "BMI": 26.0, "BP_sys": 130, "BP_dia": 82,
            "INR": 3.0, "Blood glucose at admission": 100,
            "HbA1c (last one before admission)": 5.6,
            "Are they on warfarin, or heparin before admission (YES/NO)": "YES"
        }),
        # 8) Prior stroke and TIA yes
        row({
            "Gender": "Male", "Age": 65, "BMI": 28.0, "BP_sys": 145, "BP_dia": 90,
            "Personal previous history of stoke (YES/NO)": "YES",
            "Personal previous history of Transient Ischemic Attack (YES/NO)": "YES",
            "Known case of Hypertension (YES/NO)": "YES"
        }),
        # 9) Acute interventions: fibrinolytic and thrombectomy
        row({
            "Gender": "Female", "Age": 70, "BMI": 25.0, "BP_sys": 150, "BP_dia": 93,
            "IV fibrinolytic therapy is given (YES/NO)": "YES",
            "Mechanical thrombectomy done (YES/NO)": "YES",
            "Timing of CT scan": "Early",
            "Fibrinolytic therapy timing": "Within 3h"
        }),
        # 10) Very high BP and troponin
        row({
            "Gender": "Male", "Age": 75, "BMI": 31.0, "BP_sys": 180, "BP_dia": 110,
            "Troponin level at admission": 0.25,
            "Known case of Hypertension (YES/NO)": "YES",
            "known case of coronary heart disease (YES/NO)": "YES"
        }),
        # 11) Older male with AFib on warfarin, elevated BP and INR
        row({
            "Gender": "Male", "Age": 80, "BMI": 29.5, "BP_sys": 160, "BP_dia": 95,
            "known case of atrial fibrillation (YES/NO)": "YES",
            "Are they on warfarin, or heparin before admission (YES/NO)": "YES",
            "INR": 2.7,
            "Known case of Hypertension (YES/NO)": "YES",
            "Troponin level at admission": 0.12,
            "known case of coronary heart disease (YES/NO)": "YES"
        }),
        # 12) Female with uncontrolled diabetes, very high glucose/HbA1c
        row({
            "Gender": "Female", "Age": 40, "BMI": 30.0, "BP_sys": 125, "BP_dia": 80,
            "known case of diabetes (YES/NO)": "YES",
            "Blood glucose at admission": 300,
            "HbA1c (last one before admission)": 11.2
        }),
        # 13) Male with prior stroke, well-controlled risk factors
        row({
            "Gender": "Male", "Age": 65, "BMI": 26.0, "BP_sys": 130, "BP_dia": 82,
            "Personal previous history of stoke (YES/NO)": "YES",
            "Personal previous history of Transient Ischemic Attack (YES/NO)": "NO",
            "Known case of Hypertension (YES/NO)": "YES",
            "Blood glucose at admission": 110,
            "HbA1c (last one before admission)": 6.2
        }),
        # 14) Fibrinolysis within 3h, early CT
        row({
            "Gender": "Female", "Age": 70, "BMI": 27.0, "BP_sys": 155, "BP_dia": 92,
            "IV fibrinolytic therapy is given (YES/NO)": "YES",
            "Fibrinolytic therapy timing": "Within 3h",
            "Timing of CT scan": "Early"
        }),
        # 15) High cholesterol + diabetes with high BP
        row({
            "Gender": "Male", "Age": 58, "BMI": 32.0, "BP_sys": 150, "BP_dia": 95,
            "known case of diabetes (YES/NO)": "YES",
            "Blood glucose at admission": 220,
            "HbA1c (last one before admission)": 9.0,
            "Total cholesterol (last one before admission)": 280,
            "Cholesterol level at admission": 260,
            "Known case of Hypertension (YES/NO)": "YES"
        }),
        # 16) Pre-admission anticoagulation, moderate INR
        row({
            "Gender": "Female", "Age": 62, "BMI": 24.5, "BP_sys": 140, "BP_dia": 88,
            "Are they on warfarin, or heparin before admission (YES/NO)": "YES",
            "INR": 1.5,
            "Known case of Hypertension (YES/NO)": "YES"
        }),
        # 17) Hypotension with elevated troponin and CHD
        row({
            "Gender": "Male", "Age": 55, "BMI": 27.0, "BP_sys": 85, "BP_dia": 55,
            "Troponin level at admission": 0.30,
            "known case of coronary heart disease (YES/NO)": "YES"
        }),
        # 18) Healthy young female athlete
        row({
            "Gender": "Female", "Age": 30, "BMI": 19.0, "BP_sys": 112, "BP_dia": 72,
            "Known case of Hypertension (YES/NO)": "NO",
            "known case of diabetes (YES/NO)": "NO",
            "known case of atrial fibrillation (YES/NO)": "NO",
            "Blood glucose at admission": 90,
            "HbA1c (last one before admission)": 5.1,
            "INR": 1.0
        }),
        # 19) Thrombectomy, late CT, AFib with high BP
        row({
            "Gender": "Male", "Age": 68, "BMI": 28.0, "BP_sys": 160, "BP_dia": 98,
            "Mechanical thrombectomy done (YES/NO)": "YES",
            "Timing of CT scan": "Late",
            "known case of atrial fibrillation (YES/NO)": "YES",
            "Known case of Hypertension (YES/NO)": "YES"
        }),
        # 20) Oral contraceptive use with elevated cholesterol
        row({
            "Gender": "Female", "Age": 52, "BMI": 26.0, "BP_sys": 138, "BP_dia": 86,
            "oral contraceptive use in female(YES/NO)": "YES",
            "Total cholesterol (last one before admission)": 245,
            "Cholesterol level at admission": 230
        }),
        # 21) Severe obesity with uncontrolled diabetes and hypertension
        row({
            "Gender": "Female", "Age": 35, "BMI": 40.0, "BP_sys": 160, "BP_dia": 100,
            "Known case of Hypertension (YES/NO)": "YES",
            "known case of diabetes (YES/NO)": "YES",
            "Blood glucose at admission": 250,
            "HbA1c (last one before admission)": 10.5
        }),
        # 22) Elevated troponin with normal BP, CHD
        row({
            "Gender": "Male", "Age": 72, "BMI": 24.0, "BP_sys": 120, "BP_dia": 78,
            "Troponin level at admission": 0.40,
            "known case of coronary heart disease (YES/NO)": "YES"
        }),
        # 23) Fibrinolysis early with high BP
        row({
            "Gender": "Female", "Age": 65, "BMI": 25.0, "BP_sys": 150, "BP_dia": 92,
            "Known case of Hypertension (YES/NO)": "YES",
            "IV fibrinolytic therapy is given (YES/NO)": "YES",
            "Timing of CT scan": "Early"
        }),
        # 24) Very high cholesterol with diabetes
        row({
            "Gender": "Male", "Age": 55, "BMI": 33.0, "BP_sys": 145, "BP_dia": 90,
            "known case of diabetes (YES/NO)": "YES",
            "Blood glucose at admission": 240,
            "HbA1c (last one before admission)": 9.5,
            "Total cholesterol (last one before admission)": 300,
            "Cholesterol level at admission": 280
        }),
        # 25) OCP exposure with normal vitals
        row({
            "Gender": "Female", "Age": 48, "BMI": 22.0, "BP_sys": 115, "BP_dia": 75,
            "oral contraceptive use in female(YES/NO)": "YES",
            "INR": 1.0
        }),
        # 26) AFib with thrombectomy and elevated BP
        row({
            "Gender": "Male", "Age": 70, "BMI": 27.0, "BP_sys": 155, "BP_dia": 95,
            "known case of atrial fibrillation (YES/NO)": "YES",
            "Mechanical thrombectomy done (YES/NO)": "YES"
        }),
        # 27) Hypotension with elevated troponin and CHD
        row({
            "Gender": "Female", "Age": 59, "BMI": 25.0, "BP_sys": 90, "BP_dia": 55,
            "Troponin level at admission": 0.35,
            "known case of coronary heart disease (YES/NO)": "YES"
        }),
        # 28) Warfarin with moderate INR, hypertension
        row({
            "Gender": "Male", "Age": 62, "BMI": 26.0, "BP_sys": 150, "BP_dia": 92,
            "Are they on warfarin, or heparin before admission (YES/NO)": "YES",
            "INR": 2.2,
            "Known case of Hypertension (YES/NO)": "YES"
        }),
        # 29) Low-risk male, normal vitals and labs
        row({
            "Gender": "Male", "Age": 45, "BMI": 23.0, "BP_sys": 118, "BP_dia": 76,
            "Known case of Hypertension (YES/NO)": "NO",
            "known case of diabetes (YES/NO)": "NO",
            "known case of atrial fibrillation (YES/NO)": "NO",
            "Blood glucose at admission": 95,
            "HbA1c (last one before admission)": 5.3,
            "INR": 1.0
        }),
        # 30) Elderly male with clustered comorbidities
        row({
            "Gender": "Male", "Age": 80, "BMI": 28.0, "BP_sys": 165, "BP_dia": 100,
            "Known case of Hypertension (YES/NO)": "YES",
            "known case of diabetes (YES/NO)": "YES",
            "known case of atrial fibrillation (YES/NO)": "YES",
            "Blood glucose at admission": 230,
            "HbA1c (last one before admission)": 9.2,
            "INR": 1.3
        }),
    ]

    df_cases = pd.DataFrame(cases)
    # Append extreme high-risk synthetic cases
    extreme = [
        row({
            "Gender": "Male", "Age": 88, "BMI": 35.0, "BP_sys": 200, "BP_dia": 120,
            "Known case of Hypertension (YES/NO)": "YES", "known case of diabetes (YES/NO)": "YES",
            "known case of atrial fibrillation (YES/NO)": "YES", "known case of coronary heart disease (YES/NO)": "YES",
            "Blood glucose at admission": 420, "HbA1c (last one before admission)": 13.5,
            "Troponin level at admission": 1.2, "INR": 2.8,
            "Are they on warfarin, or heparin before admission (YES/NO)": "YES"
        }),
        row({
            "Gender": "Female", "Age": 82, "BMI": 33.0, "BP_sys": 190, "BP_dia": 115,
            "Known case of Hypertension (YES/NO)": "YES", "known case of diabetes (YES/NO)": "YES",
            "known case of atrial fibrillation (YES/NO)": "YES", "known case of coronary heart disease (YES/NO)": "YES",
            "Blood glucose at admission": 380, "HbA1c (last one before admission)": 12.7,
            "Troponin level at admission": 0.95, "INR": 3.2,
            "Are they on warfarin, or heparin before admission (YES/NO)": "YES"
        }),
        row({
            "Gender": "Male", "Age": 75, "BMI": 31.0, "BP_sys": 185, "BP_dia": 110,
            "Known case of Hypertension (YES/NO)": "YES", "known case of diabetes (YES/NO)": "YES",
            "known case of atrial fibrillation (YES/NO)": "YES", "known case of coronary heart disease (YES/NO)": "YES",
            "Blood glucose at admission": 500, "HbA1c (last one before admission)": 14.0,
            "Troponin level at admission": 3.0, "INR": 4.5,
            "Are they on warfarin, or heparin before admission (YES/NO)": "YES"
        }),
        row({
            "Gender": "Female", "Age": 79, "BMI": 29.0, "BP_sys": 195, "BP_dia": 118,
            "Known case of Hypertension (YES/NO)": "YES", "known case of diabetes (YES/NO)": "YES",
            "known case of atrial fibrillation (YES/NO)": "YES", "known case of coronary heart disease (YES/NO)": "YES",
            "Blood glucose at admission": 410, "HbA1c (last one before admission)": 13.1,
            "Troponin level at admission": 1.7, "INR": 3.8,
            "Are they on warfarin, or heparin before admission (YES/NO)": "YES",
            "Personal previous history of stoke (YES/NO)": "YES",
            "Personal previous history of Transient Ischemic Attack (YES/NO)": "YES"
        })
    ]
    df_extreme = pd.DataFrame(extreme)
    df_extreme = df_extreme.reindex(columns=all_cols)
    df_cases = pd.concat([df_cases, df_extreme], ignore_index=True)
    df_cases = df_cases.reindex(columns=all_cols)
    return df_cases


def load_models():
    _ensure_custom_classes_for_unpickle()
    paths = {
        "lr": os.path.join(MODELS_DIR, "model_lr.joblib"),
        "rf": os.path.join(MODELS_DIR, "model_rf.joblib"),
        "gb": os.path.join(MODELS_DIR, "model_gb.joblib"),
        "xgb": os.path.join(MODELS_DIR, "model_xgb.joblib"),
        "lgb": os.path.join(MODELS_DIR, "model_lgb.joblib"),
    }
    models = {}
    for k, p in paths.items():
        if os.path.exists(p):
            models[k] = joblib.load(p)
    cat_model = None
    cat_path = os.path.join(MODELS_DIR, "model_cat.cbm")
    if HAS_CAT and os.path.exists(cat_path):
        m = CatBoostClassifier()
        m.load_model(cat_path)
        cat_model = m
    return models, cat_model


def main():
    # Load raw data to get columns; ensure we include all training feature bases
    df = pd.read_csv(DATA_PATH)
    tgt = detect_target_column(df)
    X_raw_cols = [c.strip() for c in df.columns if c != tgt]

    # Build test cases
    df_cases_raw = build_test_cases(X_raw_cols)
    # Optionally limit via env var
    limit = os.environ.get("TEST_CASE_COUNT")
    if limit:
        try:
            n = int(limit)
            if n > 0:
                df_cases_raw = df_cases_raw.iloc[:n].copy()
        except Exception:
            pass

    # Add a dummy target so prepare_dataset knows what to drop
    dummy_target_name = "Death outcome (YES/NO)"
    df_cases_raw[dummy_target_name] = 0
    # Prepare features using the same function as training
    X_cases, _, cat_cols, num_cols, _ = prepare_dataset(df_cases_raw)
    # Cast categorical columns to string/object to avoid mixed dtype issues in imputers
    X_cases = X_cases.copy()
    for c in cat_cols:
        X_cases[c] = X_cases[c].astype(str)

    # Load thresholds
    thr_path = os.path.join(REPORTS_DIR, "thresholds.json")
    thresholds = {}
    if os.path.exists(thr_path):
        try:
            with open(thr_path, "r", encoding="utf-8") as f:
                thresholds = json.load(f)
        except Exception:
            thresholds = {}
    # Fallback threshold
    default_thr = float(os.environ.get("OPERATING_THRESHOLD", "0.01"))

    # Load models
    models, cat_model = load_models()
    model_keys = list(models.keys()) + (["cat"] if cat_model is not None else [])
    if len(model_keys) == 0:
        raise FileNotFoundError("No saved models found in phase5b/models. Please run phase5b/pipeline.py first.")

    # Predictions across models
    results = {
        "case_id": [],
    }
    # Build descriptive labels from raw test-case rows
    case_labels = [describe_case(df_cases_raw.iloc[i]) for i in range(len(df_cases_raw))]
    results["case_id"] = case_labels[:len(X_cases)]

    # Compute probabilities for sklearn pipelines
    for k, pipe in models.items():
        prob = pipe.predict_proba(X_cases)[:, 1]
        thr = float(thresholds.get(k, default_thr))
        results[f"prob_{k}"] = prob
        results[f"pred_{k}"] = (prob >= thr).astype(int)

    # CatBoost predictions (native categorical) — tolerate failure and continue
    if cat_model is not None:
        try:
            # Align to training feature order if available
            cat_meta_path = os.path.join(MODELS_DIR, "model_cat_meta.json")
            feature_order = None
            if os.path.exists(cat_meta_path):
                try:
                    with open(cat_meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                        feature_order = meta.get("feature_order", None)
                except Exception:
                    feature_order = None
            # Default to cat_cols + num_cols if no meta
            cols_order = feature_order if isinstance(feature_order, list) and feature_order else (cat_cols + num_cols)
            X_cb = X_cases[cols_order].copy()
            for c in cat_cols:
                X_cb[c] = X_cb[c].astype(str).replace({"nan": "nan"}).fillna("nan")
            # Use numpy object array with categorical indices aligned to first columns
            X_vals = X_cb.to_numpy(dtype=object)
            # Determine categorical indices within the final order
            cat_idx = [i for i, c in enumerate(cols_order) if c in set(cat_cols)]
            pool = Pool(X_vals, cat_features=cat_idx)
            prob_cat = cat_model.predict_proba(pool)[:, 1]
            thr_cat = float(thresholds.get("cat", default_thr))
            results["prob_cat"] = prob_cat
            results["pred_cat"] = (prob_cat >= thr_cat).astype(int)
        except Exception as e:
            try:
                with open(os.path.join(REPORTS_DIR, "test_cases_cat_error.txt"), "w", encoding="utf-8") as f:
                    f.write(str(e))
            except Exception:
                pass

    # Save CSV results
    out_df = pd.DataFrame(results)
    out_csv = os.path.join(REPORTS_DIR, "phase5b_test_cases_results.csv")
    out_df.to_csv(out_csv, index=False)

    # Grouped bar chart: x = descriptive case labels, bars = model probabilities
    prob_cols = [c for c in out_df.columns if c.startswith("prob_")]
    df_long = out_df.melt(id_vars=["case_id"], value_vars=prob_cols, var_name="model", value_name="prob")
    df_long["model"] = df_long["model"].str.replace("prob_", "", regex=False)
    plt.figure(figsize=(max(12, len(out_df) * 0.6), 8))
    sns.barplot(data=df_long, x="case_id", y="prob", hue="model")
    plt.xticks(rotation=60, ha="right")
    plt.ylabel("Risk probability")
    plt.ylim(0.0, 1.0)
    plt.title("Phase5b: Risk probabilities across models for synthetic test cases")
    plt.tight_layout()
    out_svg = os.path.join(VISUALS_DIR, "phase5b_test_cases_barchart_models.svg")
    plt.savefig(out_svg, format="svg")
    plt.close()

    # Save a JSON summary with thresholds used
    thr_used = {k: float(thresholds.get(k, default_thr)) for k in model_keys}
    with open(os.path.join(REPORTS_DIR, "phase5b_test_cases_thresholds_used.json"), "w", encoding="utf-8") as f:
        json.dump(thr_used, f, indent=2)

    print(f"Saved test case comparison to {out_csv}")
    print(f"Saved bar chart to {out_svg}")


if __name__ == "__main__":
    main()