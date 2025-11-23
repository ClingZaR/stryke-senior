import os
import re
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


YES_TERMS = {"yes", "y", "true", "t", "1"}
NO_TERMS = {"no", "n", "false", "f", "0"}
NA_TERMS = {"", "nan", "na", "n/a", "not applicable", "none", "null"}


def _lower_strip(v: object) -> str:
    if pd.isna(v):
        return "nan"
    return str(v).strip().lower()


def detect_target_column(df: pd.DataFrame) -> str:
    # Prefer explicit clinical death outcome naming
    for c in df.columns:
        name = str(c).strip().lower()
        if "death" in name and "outcome" in name:
            return c
    # fallbacks
    for c in ["Death outcome (YES/NO)", "death", "outcome", "stroke", "target", "label", "y"]:
        if c in df.columns:
            return c
    return df.columns[-1]


def coerce_binary(series: pd.Series) -> pd.Series:
    if series.dtype.kind in {"i", "u", "f"}:
        # already numeric 0/1 or similar
        return (series.astype(float)).copy()
    vals = series.map(_lower_strip)
    uniq = set(vals.unique())
    pos_terms = YES_TERMS | {"positive", "pos", "death", "deceased", "stroke"}
    mapping: Dict[str, float] = {}
    for u in uniq:
        if u in NA_TERMS:
            mapping[u] = np.nan
        elif u in pos_terms:
            mapping[u] = 1.0
        elif u in NO_TERMS:
            mapping[u] = 0.0
        else:
            # unknown term: keep as NaN to avoid spurious categories
            mapping[u] = np.nan
    out = vals.map(mapping)
    return out


def likely_yes_no_column(s: pd.Series) -> bool:
    vals = s.dropna().astype(str).map(lambda x: x.strip().lower())
    uniq = set(vals.unique())
    return uniq.issubset(YES_TERMS | NO_TERMS | NA_TERMS)


def normalize_yes_no_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        s = df[col]
        if s.dtype == object:
            if "(yes/no)" in str(col).lower() or likely_yes_no_column(s):
                df[col] = coerce_binary(s)
    return df


def drop_id_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    df = df.copy()
    group = None
    for id_col in ["CMRN", "cmrn", "patient_id", "id"]:
        if id_col in df.columns:
            group = df[id_col].astype(str)
            df = df.drop(columns=[id_col])
            break
    return df, group


def parse_datetime_series(s: pd.Series) -> pd.Series:
    # Robust parser using pandas; attempts multiple formats
    s2 = s.astype(str).str.strip()
    # Replace common ‘ast’ suffix with empty
    s2 = s2.str.replace(r"\s*ast$", "", regex=True)
    # Unify AM/PM spacing variants
    s2 = s2.str.replace(r"(\d)\s*(am|pm)$", r"\1 \2", flags=re.IGNORECASE, regex=True)
    dt = pd.to_datetime(s2, errors="coerce", dayfirst=False, infer_datetime_format=True, utc=False)
    return dt


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Identify columns by fuzzy names
    name_map: Dict[str, str] = {}
    cols = {str(c).strip(): c for c in df.columns}
    for k in cols:
        kl = k.lower()
        if "admission" in kl and ("time" in kl or "timing" in kl):
            name_map.setdefault("admission", cols[k])
        if "ct" in kl and ("time" in kl or "timing" in kl):
            name_map.setdefault("ct", cols[k])
        if "fibrinolytic" in kl and ("time" in kl or "timing" in kl):
            name_map.setdefault("tpa", cols[k])

    adm_dt = parse_datetime_series(df[name_map["admission"]]) if "admission" in name_map else pd.Series(pd.NaT, index=df.index)
    ct_dt = parse_datetime_series(df[name_map["ct"]]) if "ct" in name_map else pd.Series(pd.NaT, index=df.index)
    tpa_dt = parse_datetime_series(df[name_map["tpa"]]) if "tpa" in name_map else pd.Series(pd.NaT, index=df.index)

    df["admission_dt"] = adm_dt
    df["ct_dt"] = ct_dt
    df["tpa_dt"] = tpa_dt

    def to_minutes(a: pd.Series, b: pd.Series) -> pd.Series:
        delta = (b - a).dt.total_seconds() / 60.0
        return delta

    df["admission_to_ct_minutes"] = to_minutes(adm_dt, ct_dt)
    df["admission_to_tpa_minutes"] = to_minutes(adm_dt, tpa_dt)
    df["admission_hour"] = adm_dt.dt.hour
    df["admission_weekday"] = adm_dt.dt.weekday

    return df


def compute_bmi(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Use exact column names if present; otherwise try fuzzy
    w_cols = [c for c in df.columns if str(c).strip().lower() in {"weight", "weight on admission"}]
    h_cols = [c for c in df.columns if str(c).strip().lower() in {"height"}]
    weight_col = w_cols[0] if w_cols else None
    height_col = h_cols[0] if h_cols else None
    if weight_col and height_col:
        w = pd.to_numeric(df[weight_col], errors="coerce")
        h = pd.to_numeric(df[height_col], errors="coerce")
        bmi = w / (h ** 2)
        df["BMI_computed"] = bmi
        # Prefer existing BMI if present; otherwise use computed
        bmi_cols = [c for c in df.columns if str(c).strip().lower() == "bmi"]
        if bmi_cols:
            df["BMI_final"] = df[bmi_cols[0]].where(~df[bmi_cols[0]].isna(), bmi)
        else:
            df["BMI_final"] = bmi
    return df


def derive_bp_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    sys_col = next((c for c in df.columns if str(c).strip().lower() in {"bp_sys", "systolic", "bp_systolic"}), None)
    dia_col = next((c for c in df.columns if str(c).strip().lower() in {"bp_dia", "diastolic", "bp_diastolic"}), None)
    if sys_col and dia_col:
        sys = pd.to_numeric(df[sys_col], errors="coerce")
        dia = pd.to_numeric(df[dia_col], errors="coerce")
        df["pulse_pressure"] = sys - dia
        df["map_estimate"] = dia + (sys - dia) / 3.0
    return df


def normalize_sex_conditioned(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    gender_col = next((c for c in df.columns if str(c).strip().lower() in {"gender", "sex"}), None)
    if not gender_col:
        return df
    males = df[gender_col].astype(str).str.strip().str.lower() == "male"
    for c in df.columns:
        if "oral contraceptive" in str(c).strip().lower():
            # set to 0 for males
            s = df[c]
            if s.dtype == object:
                df.loc[males, c] = "no"
            df[c] = coerce_binary(df[c])
    return df


def log_transform_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    candidates = []
    for c in df.columns:
        name = str(c).strip().lower()
        if any(k in name for k in ["troponin", "inr", "glucose", "cholesterol", "hbA1c".lower()]):
            candidates.append(c)
    for c in candidates:
        x = pd.to_numeric(df[c], errors="coerce")
        df[f"log1p_{c}"] = np.log1p(x.clip(lower=0))
    return df


DEFAULT_BOUNDS: Dict[str, Tuple[float, float]] = {
    "bp_sys": (70, 260),
    "bp_dia": (40, 150),
    "age": (0, 120),
    "bmi": (10, 60),
}


def clip_plausible_ranges(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        name = str(c).strip().lower()
        if name in DEFAULT_BOUNDS:
            lo, hi = DEFAULT_BOUNDS[name]
            df[c] = pd.to_numeric(df[c], errors="coerce").clip(lower=lo, upper=hi)
    return df


def prepare_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str], Optional[pd.Series]]:
    # Standardize columns names by stripping
    df.columns = [str(c).strip() for c in df.columns]

    # Target
    target_col = detect_target_column(df)
    y_raw = df[target_col]
    df_features = df.drop(columns=[target_col])

    # Normalize yes/no and sex-conditioned variables
    df_features = normalize_yes_no_columns(df_features)
    df_features = normalize_sex_conditioned(df_features)

    # Drop IDs but retain group labels
    df_features, groups = drop_id_columns(df_features)

    # Time features and clinical derivations
    df_features = add_time_features(df_features)
    df_features = compute_bmi(df_features)
    df_features = derive_bp_features(df_features)
    df_features = log_transform_columns(df_features)
    df_features = clip_plausible_ranges(df_features)

    # Binary target
    y = coerce_binary(y_raw)

    # Column typing
    numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df_features.columns if c not in numeric_cols]

    return df_features, y, categorical_cols, numeric_cols, groups