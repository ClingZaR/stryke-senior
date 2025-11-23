import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from matplotlib.patches import Rectangle

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
PHASE_DIR = os.path.join(PROJECT_ROOT, "phase4b")
REPORTS_DIR = os.path.join(PHASE_DIR, "reports")
VISUALS_DIR = os.path.join(PHASE_DIR, "visuals")
MODELS_DIR = os.path.join(PHASE_DIR, "models")
DATA_PATH = os.path.join(PROJECT_ROOT, "clean_data.csv")

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(VISUALS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


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
    return df.columns[-1]


def build_test_cases(all_cols: list[str]) -> pd.DataFrame:
    # Base template with NaNs for all features
    base = {c: np.nan for c in all_cols}

    def row(update: dict):
        r = base.copy()
        r.update(update)
        return r

    cases = [
        # 1) Low-risk young female, normal vitals
        row({
            "Gender": "Female", "Age": 25, "BMI": 22.0, "BP_sys": 110, "BP_dia": 70,
            "Known case of Hypertension (YES/NO)": "NO", "known case of diabetes (YES/NO)": "NO",
            "known case of atrial fibrillation (YES/NO)": "NO", "Troponin level at admission ": 0.01,
            "Blood glucose at admission ": 95, "HbA1c (last one before admission)": 5.2,
            "INR ": 1.0
        }),
        # 2) Middle-aged male with hypertension
        row({
            "Gender": "Male", "Age": 50, "BMI": 27.5, "BP_sys": 145, "BP_dia": 90,
            "Known case of Hypertension (YES/NO)": "YES", "known case of diabetes (YES/NO)": "NO",
            "known case of atrial fibrillation (YES/NO)": "NO", "Blood glucose at admission ": 105,
            "HbA1c (last one before admission)": 5.8, "INR ": 1.1
        }),
        # 3) Older male, AFib + diabetes, high glucose/HbA1c
        row({
            "Gender": "Male", "Age": 72, "BMI": 29.0, "BP_sys": 150, "BP_dia": 92,
            "known case of atrial fibrillation (YES/NO)": "YES", "Known case of Hypertension (YES/NO)": "YES",
            "known case of diabetes (YES/NO)": "YES", "Blood glucose at admission ": 210,
            "HbA1c (last one before admission)": 8.9, "Troponin level at admission ": 0.08,
            "INR ": 1.2
        }),
        # 4) Older female, oral contraceptives yes (if applicable), moderate BP
        row({
            "Gender": "Female", "Age": 68, "BMI": 24.0, "BP_sys": 135, "BP_dia": 85,
            "oral contraceptive use in female(YES/NO)": "YES", "Known case of Hypertension (YES/NO)": "YES",
            "known case of diabetes (YES/NO)": "NO", "Blood glucose at admission ": 115,
            "HbA1c (last one before admission)": 6.0
        }),
        # 5) Obese, high cholesterol
        row({
            "Gender": "Male", "Age": 60, "BMI": 34.0, "BP_sys": 140, "BP_dia": 88,
            "Total cholesterol (last one before admission)": 260,
            "Cholesterol level at admission ": 240,
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
            "INR ": 3.0, "Blood glucose at admission ": 100,
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
            "Timing of CT scan ": "Early",
            "Fibrinolytic therapy timing": "Within 3h"
        }),
        # 10) Very high BP and troponin
        row({
            "Gender": "Male", "Age": 75, "BMI": 31.0, "BP_sys": 180, "BP_dia": 110,
            "Troponin level at admission ": 0.25,
            "Known case of Hypertension (YES/NO)": "YES",
            "known case of coronary heart disease (YES/NO)": "YES"
        }),
        # 11) Older male with AFib on warfarin, elevated BP and INR
        row({
            "Gender": "Male", "Age": 80, "BMI": 29.5, "BP_sys": 160, "BP_dia": 95,
            "known case of atrial fibrillation (YES/NO)": "YES",
            "Are they on warfarin, or heparin before admission (YES/NO)": "YES",
            "INR ": 2.7,
            "Known case of Hypertension (YES/NO)": "YES",
            "Troponin level at admission ": 0.12,
            "known case of coronary heart disease (YES/NO)": "YES"
        }),
        # 12) Female with uncontrolled diabetes, very high glucose/HbA1c
        row({
            "Gender": "Female", "Age": 40, "BMI": 30.0, "BP_sys": 125, "BP_dia": 80,
            "known case of diabetes (YES/NO)": "YES",
            "Blood glucose at admission ": 300,
            "HbA1c (last one before admission)": 11.2
        }),
        # 13) Male with prior stroke, well-controlled risk factors
        row({
            "Gender": "Male", "Age": 65, "BMI": 26.0, "BP_sys": 130, "BP_dia": 82,
            "Personal previous history of stoke (YES/NO)": "YES",
            "Personal previous history of Transient Ischemic Attack (YES/NO)": "NO",
            "Known case of Hypertension (YES/NO)": "YES",
            "Blood glucose at admission ": 110,
            "HbA1c (last one before admission)": 6.2
        }),
        # 14) Fibrinolysis within 3h, early CT
        row({
            "Gender": "Female", "Age": 70, "BMI": 27.0, "BP_sys": 155, "BP_dia": 92,
            "IV fibrinolytic therapy is given (YES/NO)": "YES",
            "Fibrinolytic therapy timing": "Within 3h",
            "Timing of CT scan ": "Early"
        }),
        # 15) High cholesterol + diabetes with high BP
        row({
            "Gender": "Male", "Age": 58, "BMI": 32.0, "BP_sys": 150, "BP_dia": 95,
            "known case of diabetes (YES/NO)": "YES",
            "Blood glucose at admission ": 220,
            "HbA1c (last one before admission)": 9.0,
            "Total cholesterol (last one before admission)": 280,
            "Cholesterol level at admission ": 260,
            "Known case of Hypertension (YES/NO)": "YES"
        }),
        # 16) Pre-admission anticoagulation, moderate INR
        row({
            "Gender": "Female", "Age": 62, "BMI": 24.5, "BP_sys": 140, "BP_dia": 88,
            "Are they on warfarin, or heparin before admission (YES/NO)": "YES",
            "INR ": 1.5,
            "Known case of Hypertension (YES/NO)": "YES"
        }),
        # 17) Hypotension with elevated troponin and CHD
        row({
            "Gender": "Male", "Age": 55, "BMI": 27.0, "BP_sys": 85, "BP_dia": 55,
            "Troponin level at admission ": 0.30,
            "known case of coronary heart disease (YES/NO)": "YES"
        }),
        # 18) Healthy young female athlete
        row({
            "Gender": "Female", "Age": 30, "BMI": 19.0, "BP_sys": 112, "BP_dia": 72,
            "Known case of Hypertension (YES/NO)": "NO",
            "known case of diabetes (YES/NO)": "NO",
            "known case of atrial fibrillation (YES/NO)": "NO",
            "Blood glucose at admission ": 90,
            "HbA1c (last one before admission)": 5.1,
            "INR ": 1.0
        }),
        # 19) Thrombectomy, late CT, AFib with high BP
        row({
            "Gender": "Male", "Age": 68, "BMI": 28.0, "BP_sys": 160, "BP_dia": 98,
            "Mechanical thrombectomy done (YES/NO)": "YES",
            "Timing of CT scan ": "Late",
            "known case of atrial fibrillation (YES/NO)": "YES",
            "Known case of Hypertension (YES/NO)": "YES"
        }),
        # 20) Oral contraceptive use with elevated cholesterol
        row({
            "Gender": "Female", "Age": 52, "BMI": 26.0, "BP_sys": 138, "BP_dia": 86,
            "oral contraceptive use in female(YES/NO)": "YES",
            "Total cholesterol (last one before admission)": 245,
            "Cholesterol level at admission ": 230
        }),
        # 21) Severe obesity with uncontrolled diabetes and hypertension
        row({
            "Gender": "Female", "Age": 35, "BMI": 40.0, "BP_sys": 160, "BP_dia": 100,
            "Known case of Hypertension (YES/NO)": "YES",
            "known case of diabetes (YES/NO)": "YES",
            "Blood glucose at admission ": 250,
            "HbA1c (last one before admission)": 10.5
        }),
        # 22) Elevated troponin with normal BP, CHD
        row({
            "Gender": "Male", "Age": 72, "BMI": 24.0, "BP_sys": 120, "BP_dia": 78,
            "Troponin level at admission ": 0.40,
            "known case of coronary heart disease (YES/NO)": "YES"
        }),
        # 23) Fibrinolysis early with high BP
        row({
            "Gender": "Female", "Age": 65, "BMI": 25.0, "BP_sys": 150, "BP_dia": 92,
            "Known case of Hypertension (YES/NO)": "YES",
            "IV fibrinolytic therapy is given (YES/NO)": "YES",
            "Timing of CT scan ": "Early"
        }),
        # 24) Very high cholesterol with diabetes
        row({
            "Gender": "Male", "Age": 55, "BMI": 33.0, "BP_sys": 145, "BP_dia": 90,
            "known case of diabetes (YES/NO)": "YES",
            "Blood glucose at admission ": 240,
            "HbA1c (last one before admission)": 9.5,
            "Total cholesterol (last one before admission)": 300,
            "Cholesterol level at admission ": 280
        }),
        # 25) OCP exposure with normal vitals
        row({
            "Gender": "Female", "Age": 48, "BMI": 22.0, "BP_sys": 115, "BP_dia": 75,
            "oral contraceptive use in female(YES/NO)": "YES",
            "INR ": 1.0
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
            "Troponin level at admission ": 0.35,
            "known case of coronary heart disease (YES/NO)": "YES"
        }),
        # 28) Warfarin with moderate INR, hypertension
        row({
            "Gender": "Male", "Age": 62, "BMI": 26.0, "BP_sys": 150, "BP_dia": 92,
            "Are they on warfarin, or heparin before admission (YES/NO)": "YES",
            "INR ": 2.2,
            "Known case of Hypertension (YES/NO)": "YES"
        }),
        # 29) Low-risk male, normal vitals and labs
        row({
            "Gender": "Male", "Age": 45, "BMI": 23.0, "BP_sys": 118, "BP_dia": 76,
            "Known case of Hypertension (YES/NO)": "NO",
            "known case of diabetes (YES/NO)": "NO",
            "known case of atrial fibrillation (YES/NO)": "NO",
            "Blood glucose at admission ": 95,
            "HbA1c (last one before admission)": 5.3,
            "INR ": 1.0
        }),
        # 30) Elderly male with clustered comorbidities
        row({
            "Gender": "Male", "Age": 80, "BMI": 28.0, "BP_sys": 165, "BP_dia": 100,
            "Known case of Hypertension (YES/NO)": "YES",
            "known case of diabetes (YES/NO)": "YES",
            "known case of atrial fibrillation (YES/NO)": "YES",
            "Blood glucose at admission ": 230,
            "HbA1c (last one before admission)": 9.2,
            "INR ": 1.3
        }),
    ]

    df_cases = pd.DataFrame(cases)
    # Append extremely high-risk synthetic cases
    extreme = [
        row({
            "Gender": "Male", "Age": 88, "BMI": 35.0, "BP_sys": 200, "BP_dia": 120,
            "Known case of Hypertension (YES/NO)": "YES", "known case of diabetes (YES/NO)": "YES",
            "known case of atrial fibrillation (YES/NO)": "YES", "known case of coronary heart disease (YES/NO)": "YES",
            "Blood glucose at admission ": 420, "HbA1c (last one before admission)": 13.5,
            "Troponin level at admission ": 1.2, "INR ": 2.8,
            "Are they on warfarin, or heparin before admission (YES/NO)": "YES"
        }),
        row({
            "Gender": "Female", "Age": 82, "BMI": 33.0, "BP_sys": 190, "BP_dia": 115,
            "Known case of Hypertension (YES/NO)": "YES", "known case of diabetes (YES/NO)": "YES",
            "known case of atrial fibrillation (YES/NO)": "YES", "known case of coronary heart disease (YES/NO)": "YES",
            "Blood glucose at admission ": 380, "HbA1c (last one before admission)": 12.7,
            "Troponin level at admission ": 0.95, "INR ": 3.2,
            "Are they on warfarin, or heparin before admission (YES/NO)": "YES"
        }),
        row({
            "Gender": "Male", "Age": 75, "BMI": 31.0, "BP_sys": 185, "BP_dia": 110,
            "Known case of Hypertension (YES/NO)": "YES", "known case of diabetes (YES/NO)": "YES",
            "known case of atrial fibrillation (YES/NO)": "YES", "known case of coronary heart disease (YES/NO)": "YES",
            "Blood glucose at admission ": 500, "HbA1c (last one before admission)": 14.0,
            "Troponin level at admission ": 3.0, "INR ": 4.5,
            "Are they on warfarin, or heparin before admission (YES/NO)": "YES"
        }),
        row({
            "Gender": "Female", "Age": 79, "BMI": 29.0, "BP_sys": 195, "BP_dia": 118,
            "Known case of Hypertension (YES/NO)": "YES", "known case of diabetes (YES/NO)": "YES",
            "known case of atrial fibrillation (YES/NO)": "YES", "known case of coronary heart disease (YES/NO)": "YES",
            "Blood glucose at admission ": 410, "HbA1c (last one before admission)": 13.1,
            "Troponin level at admission ": 1.7, "INR ": 3.8,
            "Are they on warfarin, or heparin before admission (YES/NO)": "YES",
            "Personal previous history of stoke (YES/NO)": "YES",
            "Personal previous history of Transient Ischemic Attack (YES/NO)": "YES"
        })
    ]
    df_extreme = pd.DataFrame(extreme)
    df_extreme = df_extreme.reindex(columns=all_cols)
    df_cases = pd.concat([df_cases, df_extreme], ignore_index=True)
    # ensure columns order matches training X columns, fill missing
    df_cases = df_cases.reindex(columns=all_cols)
    return df_cases


def main():
    # Load training data to get feature columns
    df = pd.read_csv(DATA_PATH)
    target_col = detect_target_column(df)
    X_cols = [c for c in df.columns if c != target_col]

    # Build test cases with comprehensive parameters
    df_cases = build_test_cases(X_cols)
    # Optionally limit the number of synthetic test cases via env var
    _limit = os.environ.get("TEST_CASE_COUNT")
    if _limit is not None:
        try:
            _n = int(_limit)
            if _n > 0:
                df_cases = df_cases.iloc[:_n].copy()
        except Exception:
            pass

    # Load model and threshold
    model_path = os.path.join(MODELS_DIR, "phase4b_best_calibrated_model.joblib")
    meta_path = os.path.join(MODELS_DIR, "phase4b_best_model_meta.json")
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        raise FileNotFoundError("Best calibrated model or metadata not found. Please run phase4b/pipeline.py first.")
    clf = joblib.load(model_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    thr = float(meta.get("threshold", 0.5))
    # Clinically sensible override: allow high-recall operating mode or explicit threshold via env vars
    thr_override = os.environ.get("OPERATING_THRESHOLD")
    goal = os.environ.get("OPERATING_GOAL")
    if thr_override is not None:
        try:
            thr = float(thr_override)
        except Exception:
            pass
    elif goal == "high_recall":
        sweep_path = os.path.join(REPORTS_DIR, "none_threshold_sweep.csv")
        if os.path.exists(sweep_path):
            try:
                sweep = pd.read_csv(sweep_path)
                # Prefer thresholds with high recall; choose the one that maximizes precision then F1 among recall >= 0.7
                candidates = sweep[sweep["recall"] >= 0.7]
                if len(candidates) == 0:
                    candidates = sweep[sweep["recall"] >= 0.5]
                if len(candidates) > 0:
                    best_row = candidates.sort_values(["precision", "f1"], ascending=[False, False]).iloc[0]
                    thr = float(best_row["threshold"])
            except Exception:
                pass

    # Predict calibrated probabilities and labels
    probs = clf.predict_proba(df_cases)[:, 1]
    preds = (probs >= thr).astype(int)

    # Save results
    out_df = df_cases.copy()
    labels = [
        "Low risk",
        "Middle-aged hypertensive",
        "Old",
        "Old female",
        "High cholesterol",
        "Underweight",
        "Anticoagulated",
        "Young diabetic",
        "Coronary disease",
        "Young healthy",
        "High troponin",
        "Severe diabetes",
        "Prior stroke",
        "Fibrinolysis early CT",
        "Extreme risk 1",
        "Extreme risk 2",
        "Extreme risk 3",
        "Extreme risk 4",
    ]
    if len(labels) < len(out_df):
        labels.extend([f"Case {i+1}" for i in range(len(labels), len(out_df))])
    out_df.insert(0, "case_id", labels[:len(out_df)])
    out_df["risk_prob"] = probs
    out_df["predicted_label"] = preds
    out_csv = os.path.join(REPORTS_DIR, "phase4b_test_cases_results.csv")
    out_df.to_csv(out_csv, index=False)
    # Also save a threshold-specific copy for side-by-side comparisons
    out_csv_thr = os.path.join(REPORTS_DIR, f"phase4b_test_cases_results_threshold_{thr:.2f}.csv")
    out_df.to_csv(out_csv_thr, index=False)

    # Compute simple per-case feature attributions by one-at-a-time baseline replacement
    # Baseline is median for numeric columns, mode for categorical
    baselines = {}
    for c in X_cols:
        s = df[c]
        try:
            if pd.api.types.is_numeric_dtype(s):
                baselines[c] = s.median()
            else:
                mode = s.mode(dropna=True)
                baselines[c] = mode.iloc[0] if len(mode) > 0 else np.nan
        except Exception:
            baselines[c] = np.nan

    attrib_rows = []
    for i in range(len(out_df)):
        case_id = out_df.loc[i, "case_id"]
        X_case = df_cases.iloc[[i]].copy()
        base_prob = probs[i]
        for c in X_case.columns:
            orig_val = X_case.iloc[0][c]
            base_val = baselines.get(c, np.nan)
            # Only attempt if original is not NaN and baseline is not NaN
            if pd.isna(orig_val) or pd.isna(base_val):
                continue
            X_pert = X_case.copy()
            X_pert.iloc[0][c] = base_val
            try:
                new_prob = float(clf.predict_proba(X_pert)[:, 1][0])
            except Exception:
                # If prediction fails due to dtype mismatch, skip
                continue
            delta = base_prob - new_prob
            attrib_rows.append({
                "case_id": case_id,
                "feature": c,
                "original_value": orig_val,
                "baseline_value": base_val,
                "delta_prob": delta,
                "abs_delta": abs(delta)
            })

    attrib_df = pd.DataFrame(attrib_rows)
    attrib_out = os.path.join(REPORTS_DIR, "phase4b_test_cases_feature_attributions.csv")
    attrib_df.to_csv(attrib_out, index=False)
    attrib_out_thr = os.path.join(REPORTS_DIR, f"phase4b_test_cases_feature_attributions_threshold_{thr:.2f}.csv")
    attrib_df.to_csv(attrib_out_thr, index=False)

    # Simple summary print: top 3 contributors per case
    try:
        for case_id, g in attrib_df.groupby("case_id"):
            top3 = g.sort_values("abs_delta", ascending=False).head(3)
            print(f"Top contributors for {case_id}:")
            for _, r in top3.iterrows():
                sign = "+" if r["delta_prob"] > 0 else "-"
                print(f"  {r['feature']} ({sign}{abs(r['delta_prob']):.3f})")
    except Exception:
        pass

    # Simple visual: heatmap of risk probabilities
    plt.figure(figsize=(6, 4))
    # Define risk band thresholds
    low_thr = float(os.environ.get("LOW_RISK_THRESHOLD", "0.15"))
    bands = ["HIGH" if p >= thr else ("LOW" if p < low_thr else "INDET") for p in probs]
    heat_df = pd.DataFrame({"risk_prob": probs}, index=out_df["case_id"]) 
    annot_df = pd.DataFrame({"risk_prob": [f"{p:.2f} " + ("✓ " if p >= thr else "") + band for p, band in zip(probs, bands)]}, index=out_df["case_id"]) 
    ax = sns.heatmap(heat_df, annot=annot_df, cmap="Reds", vmin=0.0, vmax=1.0, cbar=True, fmt="", linewidths=0.5, linecolor="#eee")
    # Highlight positives with a bold rectangle
    for row_idx, p in enumerate(probs):
        if p >= thr:
            ax.add_patch(Rectangle((0, row_idx), 1, 1, fill=False, edgecolor='black', linewidth=2))
    plt.title(f"Phase 4b: Risk probability for {len(out_df)} synthetic test cases\n(threshold = {thr:.2f}; ✓ = predicted positive; bands: LOW<{low_thr:.2f}<INDET<{thr:.2f}<HIGH)")
    plt.tight_layout()
    out_svg = os.path.join(VISUALS_DIR, "phase4b_test_cases_heatmap.svg")
    plt.savefig(out_svg, format="svg")
    # Also save a threshold-specific copy for side-by-side comparisons
    out_svg_thr = os.path.join(VISUALS_DIR, f"phase4b_test_cases_heatmap_threshold_{thr:.2f}.svg")
    plt.savefig(out_svg_thr, format="svg")
    plt.close()

    # Save risk band assignments and policy
    bands_csv = os.path.join(REPORTS_DIR, f"phase4b_test_cases_risk_bands_threshold_{thr:.2f}.csv")
    bands_df = pd.DataFrame({
        "case_id": out_df["case_id"],
        "risk_prob": probs,
        "band": bands,
        "low_threshold": low_thr,
        "high_threshold": thr,
    })
    bands_df.to_csv(bands_csv, index=False)
    policy_json = os.path.join(REPORTS_DIR, "phase4b_risk_policy.json")
    with open(policy_json, "w") as f:
        json.dump({"low_threshold": low_thr, "high_threshold": thr, "bands": ["LOW","INDET","HIGH"], "notes": "LOW: below low_threshold; INDET: between low_threshold and high_threshold; HIGH: at or above high_threshold"}, f, indent=2)

    print(f"Saved test case results to {out_csv}")
    print(f"Saved threshold-specific results to {out_csv_thr}")
    print(f"Saved visual to {out_svg}")
    print(f"Saved threshold-specific visual to {out_svg_thr}")
    print(f"Saved risk band assignments to {bands_csv}")
    print(f"Saved risk policy config to {policy_json}")


class FeatureClipper(BaseEstimator, TransformerMixin):
    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper
    def fit(self, X, y=None):
        df = pd.DataFrame(X)
        self.mins_ = df.quantile(self.lower, axis=0)
        self.maxs_ = df.quantile(self.upper, axis=0)
        return self
    def transform(self, X):
        df = pd.DataFrame(X).copy()
        num_cols = [i for i, dtype in enumerate(df.dtypes) if np.issubdtype(dtype, np.number)]
        if hasattr(self, "mins_"):
            for idx in num_cols:
                col = df.columns[idx]
                df[col] = np.clip(df[col], self.mins_[col], self.maxs_[col])
        return df.to_numpy()


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


if __name__ == "__main__":
    main()