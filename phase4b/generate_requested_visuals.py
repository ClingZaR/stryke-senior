import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import importlib.util
import sys

BASE = os.path.dirname(__file__)
REPORTS = os.path.join(BASE, 'reports')
VIS = os.path.join(BASE, 'visuals')
os.makedirs(VIS, exist_ok=True)
os.makedirs(REPORTS, exist_ok=True)

sns.set(context='notebook', style='whitegrid', font_scale=1.1)


def _savefig(path: str):
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()


def load_csv(name: str) -> pd.DataFrame:
    path = os.path.join(REPORTS, name)
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df


def _ensure_custom_classes_for_unpickle():
    """Ensure custom transformers used in pickled pipelines are available on __main__.
    This avoids AttributeError when joblib loads pipelines saved with __main__.AllMissingDropper.
    """
    try:
        pipeline_path = os.path.join(BASE, 'pipeline.py')
        spec = importlib.util.spec_from_file_location('phase4b_pipeline', pipeline_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # Register on the actual __main__ module, since pickled objects reference __main__.ClassName
        main_mod = sys.modules.get('__main__') or sys.modules.get(__name__)
        for cls_name in ['AllMissingDropper', 'FeatureClipper']:
            if hasattr(mod, cls_name) and main_mod is not None:
                setattr(main_mod, cls_name, getattr(mod, cls_name))
    except Exception:
        pass


def _scatter_from_hpo(df: pd.DataFrame, sampler_kind: str, strat_col: str, neighbors_col: str, out_name: str):
    if df.empty:
        return
    metric_cols = [c for c in df.columns if c == 'mean_test_pr_auc']
    if not metric_cols:
        return
    # Drop rows without metric or strategy
    work = df[[strat_col, neighbors_col, 'mean_test_pr_auc']].copy() if neighbors_col in df.columns else df[[strat_col, 'mean_test_pr_auc']].copy()
    work = work.replace([np.inf, -np.inf], np.nan).dropna()
    if work.empty:
        return

    plt.figure(figsize=(6.2, 4.6))
    if neighbors_col in work.columns:
        sizes = np.interp(work[neighbors_col], (work[neighbors_col].min(), work[neighbors_col].max()), (40, 140))
    else:
        sizes = 80
    sc = plt.scatter(work[strat_col], work['mean_test_pr_auc'], c=work['mean_test_pr_auc'], s=sizes, cmap='viridis', alpha=0.85, edgecolor='k', linewidth=0.4)
    cbar = plt.colorbar(sc)
    cbar.set_label('mean_test_pr_auc')
    plt.xlabel('sampling_strategy')
    plt.ylabel('mean_test_pr_auc')
    title = f'HPO sweep: {sampler_kind}\nEffect of sampling intensity on PR-AUC'
    if neighbors_col in work.columns:
        title += f' (point size ~ {neighbors_col})'
    plt.title(title)
    # vertical guide lines for typical caps (0.2, 0.25, 1.0)
    for x, ls, label in [(0.1, ':', '0.1'), (0.2, '--', '0.2'), (0.25, '--', '0.25'), (1.0, ':', '1.0 (full balance)')]:
        plt.axvline(x, color='gray', linestyle=ls, alpha=0.35)
        plt.text(x, plt.ylim()[1], label, rotation=90, va='top', ha='right', color='gray', alpha=0.6, fontsize=9)
    _savefig(os.path.join(VIS, out_name))


def gen_hpo_scatter():
    smn = load_csv('smote_nearmiss_hpo_cv_results.csv')
    adn = load_csv('adasyn_nearmiss_hpo_cv_results.csv')
    if not smn.empty:
        _scatter_from_hpo(
            df=smn,
            sampler_kind='SMOTE+NearMiss',
            strat_col='param_sampler_smote__sampling_strategy',
            neighbors_col='param_sampler_smote__k_neighbors',
            out_name='hpo_scatter_smote_nearmiss.svg',
        )
    if not adn.empty:
        _scatter_from_hpo(
            df=adn,
            sampler_kind='ADASYN+NearMiss',
            strat_col='param_sampler_adasyn__sampling_strategy',
            neighbors_col='param_sampler_adasyn__n_neighbors',
            out_name='hpo_scatter_adasyn_nearmiss.svg',
        )


def gen_sampler_gate_scatter():
    comp = load_csv('phase4b_sampler_comparison.csv')
    if comp.empty:
        return
    # Normalize columns names we expect
    cols_needed = ['sampler', 'test_f1', 'test_pr_auc', 'mean_gap_f1', 'mean_gap_auc']
    if not all(c in comp.columns for c in cols_needed):
        return
    plt.figure(figsize=(7.2, 4.8))
    ax = plt.gca()
    palette = sns.color_palette('tab10')
    for i, (name, g) in enumerate(comp.groupby('sampler')):
        ax.scatter(g['mean_gap_f1'], g['test_f1'], s=90, alpha=0.9, label=name, edgecolor='k', linewidth=0.4, color=palette[i % len(palette)])
    # gates
    max_gap_f1 = 0.12
    ax.axvline(max_gap_f1, color='crimson', linestyle='--', alpha=0.6, label=f'gap_f1 gate {max_gap_f1}')
    ax.set_xlabel('Overfitting gap (F1, train - CV)')
    ax.set_ylabel('Test F1')
    ax.set_title('Sampler comparison vs Overfitting gate (lower gap is better)')
    ax.legend(ncol=2, fontsize=9)
    _savefig(os.path.join(VIS, 'sampler_overfitting_gate_f1.svg'))

    # AUC variant
    plt.figure(figsize=(7.2, 4.8))
    ax = plt.gca()
    for i, (name, g) in enumerate(comp.groupby('sampler')):
        ax.scatter(g['mean_gap_auc'], g['test_pr_auc'], s=90, alpha=0.9, label=name, edgecolor='k', linewidth=0.4, color=palette[i % len(palette)])
    max_gap_auc = 0.06
    ax.axvline(max_gap_auc, color='crimson', linestyle='--', alpha=0.6, label=f'gap_auc gate {max_gap_auc}')
    ax.set_xlabel('Overfitting gap (PR-AUC, train - CV)')
    ax.set_ylabel('Test PR-AUC')
    ax.set_title('Sampler comparison vs Overfitting gate (lower gap is better)')
    ax.legend(ncol=2, fontsize=9)
    _savefig(os.path.join(VIS, 'sampler_overfitting_gate_auc.svg'))


# -------------------- NEW: Layperson-friendly visuals --------------------

def _detect_target_and_counts():
    # Try to infer target from raw CSV
    raw = os.path.join(BASE, '..', 'clean_data.csv')
    if os.path.exists(raw):
        try:
            df = pd.read_csv(raw)
            cols = [c for c in df.columns]
            # heuristics: prefer death binary/outcome
            candidates = [c for c in cols if 'death' in c.lower()]
            if not candidates:
                candidates = [c for c in cols if 'outcome' in c.lower()]
            target = candidates[0] if candidates else None
            if target is not None:
                s = df[target]
                # map to binary
                def to_bin(v):
                    if pd.isna(v):
                        return np.nan
                    t = str(v).strip().lower()
                    if t in ['yes','y','1','true','t']:
                        return 1
                    if t in ['no','n','0','false','f']:
                        return 0
                    try:
                        return int(float(t))
                    except Exception:
                        return np.nan
                b = s.map(to_bin)
                if b.dropna().nunique() == 2:
                    counts = b.value_counts().to_dict()
                    pos = int(sorted(counts.keys())[-1])
                    neg = int(sorted(counts.keys())[0])
                    return counts.get(pos, 0), counts.get(neg, 0)
        except Exception:
            pass
    # fallback: unknown
    return None, None


def gen_class_balance_pie():
    pos, neg = _detect_target_and_counts()
    if pos is None:
        return
    total = pos + neg
    if total == 0:
        return
    pct_pos = pos / total
    labels = [f'No event ({neg})', f'Event ({pos})']
    colors = ['#8BC34A', '#E53935']
    plt.figure(figsize=(5.6, 5.0))
    plt.title('How imbalanced is our data?')
    plt.pie([neg, pos], labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, textprops={'fontsize': 11})
    plt.annotate('Only a small fraction are events.\nThis is why augmentation was explored.', xy=(0, -1.2), xytext=(0, -1.2), ha='center', fontsize=11)
    _savefig(os.path.join(VIS, 'layperson_class_balance.svg'))


def _load_best_sampling_caps():
    caps = {}
    # Try read best params JSON for representative samplers
    for name, key in [('smote_nearmiss_best_params.json', ['sampler_smote__sampling_strategy']),
                      ('adasyn_nearmiss_best_params.json', ['sampler_adasyn__sampling_strategy'])]:
        p = os.path.join(REPORTS, name)
        if os.path.exists(p):
            try:
                with open(p, 'r') as f:
                    d = json.load(f)
                for k in key:
                    if k in d:
                        caps[k] = float(d[k])
            except Exception:
                pass
    return caps


def gen_augmentation_cap_bar():
    pos, neg = _detect_target_and_counts()
    if pos is None or (pos + neg) == 0:
        return
    base_ratio = pos / neg if neg else 0
    caps = _load_best_sampling_caps()
    # defaults if missing
    smote_cap = caps.get('sampler_smote__sampling_strategy', 0.25)
    adasyn_cap = caps.get('sampler_adasyn__sampling_strategy', 0.2)
    # Convert sampling_strategy (minority/majority) to final class proportion
    def strat_to_prop(r):
        return r / (1.0 + r)
    props = {
        'Original data': pos / (pos + neg),
        'Cap with SMOTE': strat_to_prop(smote_cap),
        'Cap with ADASYN': strat_to_prop(adasyn_cap),
    }
    plt.figure(figsize=(7.2, 4.4))
    sns.barplot(x=list(props.keys()), y=list(props.values()), palette=['#90CAF9', '#FFF59D', '#FFE082'])
    plt.ylim(0, max(0.5, max(props.values()) * 1.2))
    plt.ylabel('Minority share after augmentation')
    plt.title('We cap synthetic data so minority never exceeds ~20%')
    for i, v in enumerate(props.values()):
        plt.text(i, v + 0.02, f'{v*100:.1f}%', ha='center', fontsize=11)
    _savefig(os.path.join(VIS, 'layperson_augmentation_cap.svg'))


def gen_model_leaderboard():
    comp = load_csv('phase4b_sampler_comparison.csv')
    if comp.empty:
        return
    comp = comp.copy()
    comp = comp[['sampler','test_f1','test_pr_auc','mean_gap_f1','mean_gap_auc']]
    # Pass/fail safety gates
    comp['passes'] = (comp['mean_gap_f1'] <= 0.12) & (comp['mean_gap_auc'] <= 0.06)
    comp = comp.sort_values('test_f1', ascending=False)
    plt.figure(figsize=(7.5, 4.8))
    colors = comp['passes'].map({True: '#66BB6A', False: '#B0BEC5'}).values
    sns.barplot(data=comp, x='sampler', y='test_f1', palette=colors)
    plt.ylabel('Test F1 (higher is better)')
    plt.title('Model leaderboard — green bars pass safety checks (low overfitting)')
    plt.xticks(rotation=20, ha='right')
    for i, r in enumerate(comp.itertuples(index=False)):
        plt.text(i, r.test_f1 + 0.015, f"PR-AUC {r.test_pr_auc:.2f}\n{'PASS' if r.passes else 'HOLD'}", ha='center', fontsize=9)
    _savefig(os.path.join(VIS, 'layperson_model_leaderboard.svg'))


def gen_threshold_explainer():
    summ_path = os.path.join(REPORTS, 'phase4b_summary.json')
    if not os.path.exists(summ_path):
        return
    with open(summ_path, 'r') as f:
        summ = json.load(f)
    best = summ.get('best_sampler', 'none')
    thr = summ.get('best_threshold', 0.5)
    sweep_name = f"{best}_threshold_sweep.csv"
    sweep = load_csv(sweep_name)
    if sweep.empty:
        return
    plt.figure(figsize=(7.2, 4.6))
    plt.plot(sweep['threshold'], sweep['precision'], label='Precision', color='#42A5F5')
    plt.plot(sweep['threshold'], sweep['recall'], label='Recall', color='#EF5350')
    plt.axvline(thr, color='black', linestyle='--', alpha=0.8, label=f'Chosen threshold = {thr:.2f}')
    plt.xlabel('Decision threshold')
    plt.ylabel('Score')
    plt.title('How we set the alert threshold (trade-off between catching cases and false alarms)')
    plt.legend()
    _savefig(os.path.join(VIS, 'layperson_threshold_explainer.svg'))


# -------------------- NEW: Clinical decision curve --------------------

def _get_prevalence():
    pos, neg = _detect_target_and_counts()
    if pos is None:
        return None
    total = pos + neg
    if total == 0:
        return None
    return pos / total


def gen_decision_curve():
    # Use best sampler sweep to compute net benefit curves
    summ_path = os.path.join(REPORTS, 'phase4b_summary.json')
    prev = _get_prevalence()
    if prev is None or not os.path.exists(summ_path):
        return
    with open(summ_path, 'r') as f:
        summ = json.load(f)
    best = summ.get('best_sampler', 'none')
    thr = float(summ.get('best_threshold', 0.5))
    sweep_name = f"{best}_threshold_sweep.csv"
    sweep = load_csv(sweep_name)
    if sweep.empty:
        return
    # Expect threshold, precision, recall
    if not all(c in sweep.columns for c in ['threshold','precision','recall']):
        return
    df = sweep[['threshold','precision','recall']].copy().replace([np.inf,-np.inf], np.nan).dropna()
    # Net benefit for model: NB = (TP/N) - (FP/N)*(pt/(1-pt))
    # TP/N = recall * prevalence; FP/N = (TP/N) * (1 - precision) / precision
    pt = df['threshold'].clip(1e-6, 1-1e-6)
    tp_rate = df['recall'] * prev
    fp_rate = tp_rate * (1 - df['precision']) / df['precision']
    nb_model = tp_rate - fp_rate * (pt / (1 - pt))
    # Treat-all and treat-none baselines
    nb_all = prev - (1 - prev) * (pt / (1 - pt))
    nb_none = np.zeros_like(nb_model)
    out = pd.DataFrame({
        'threshold': df['threshold'],
        'precision': df['precision'],
        'recall': df['recall'],
        'net_benefit_model': nb_model,
        'net_benefit_treat_all': nb_all,
        'net_benefit_treat_none': nb_none,
    })
    out_path = os.path.join(REPORTS, 'phase4b_decision_curve.csv')
    out.to_csv(out_path, index=False)

    plt.figure(figsize=(7.6, 4.8))
    plt.plot(out['threshold'], out['net_benefit_model'], label='Model', color='#1E88E5')
    plt.plot(out['threshold'], out['net_benefit_treat_all'], label='Treat-all', color='#8E24AA', linestyle='--')
    plt.plot(out['threshold'], out['net_benefit_treat_none'], label='Treat-none', color='#616161', linestyle=':')
    plt.axvline(thr, color='black', linestyle='--', alpha=0.8, label=f'Chosen threshold = {thr:.2f}')
    plt.xlabel('Risk threshold (probability)')
    plt.ylabel('Net benefit')
    plt.title('Clinical decision curve analysis')
    plt.legend()
    _savefig(os.path.join(VIS, 'decision_curve.svg'))


# -------------------- NEW: Subgroup fairness metrics --------------------

def _to_binary_series(s: pd.Series) -> pd.Series:
    def to_bin(v):
        if pd.isna(v):
            return np.nan
        t = str(v).strip().lower()
        if t in ['yes','y','1','true','t']:
            return 1
        if t in ['no','n','0','false','f']:
            return 0
        try:
            return int(float(t))
        except Exception:
            return np.nan
    return s.map(to_bin)


def gen_shap_explanations():
    # Load model and raw data
    model_path = os.path.join(BASE, 'models', 'phase4b_best_calibrated_model.joblib')
    meta_path = os.path.join(BASE, 'models', 'phase4b_best_model_meta.json')
    data_path = os.path.join(BASE, '..', 'clean_data.csv')
    if not (os.path.exists(model_path) and os.path.exists(meta_path) and os.path.exists(data_path)):
        return
    _ensure_custom_classes_for_unpickle()
    clf = joblib.load(model_path)
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    df = pd.read_csv(data_path)
    # Detect target
    candidates = [c for c in df.columns if 'death' in c.lower()] or [c for c in df.columns if 'outcome' in c.lower()]
    if not candidates:
        return
    target = candidates[0]
    y = _to_binary_series(df[target])
    work = df.copy()
    work['__y__'] = y
    work = work.dropna(subset=['__y__'])
    y_bin = work['__y__'].astype(int)
    X = work[[c for c in work.columns if c != target and c != '__y__']]

    # Access base estimator pipeline from the first calibrated classifier
    try:
        calibrated = clf.calibrated_classifiers_[0]
        base = getattr(calibrated, 'base_estimator', None) or getattr(calibrated, 'estimator', None)
    except Exception:
        base = None
    # Fallback: some sklearn versions keep the original estimator on CalibratedClassifierCV
    if base is None:
        base = getattr(clf, 'estimator', None)
    if base is None or not hasattr(base, 'named_steps'):
        return
    pre = base.named_steps.get('pre')
    est = base.named_steps.get('rf') or base.named_steps.get('brf')
    if pre is None or est is None:
        return

    # Transform features using fitted preprocessor
    try:
        Xt = pre.transform(X)
    except Exception:
        # If transform fails, try fit_transform as fallback (shouldn't happen for fitted pre)
        try:
            Xt = pre.fit_transform(X, y_bin)
        except Exception:
            return
    # Densify transformed matrix for SHAP
    if hasattr(Xt, 'toarray'):
        Xt = Xt.toarray()
    else:
        Xt = np.asarray(Xt)
    # Feature names (fallback to indices if not available)
    try:
        feat_names = list(pre.get_feature_names_out())
    except Exception:
        feat_names = [f'f{i}' for i in range(Xt.shape[1])]
    # Ensure feature names length aligns with transformed features
    if len(feat_names) != Xt.shape[1]:
        feat_names = [f'f{i}' for i in range(Xt.shape[1])]

    # Compute SHAP values using TreeExplainer for the forest
    try:
        explainer = shap.TreeExplainer(est)
        shap_vals = explainer.shap_values(Xt)
        expected = explainer.expected_value
    except Exception:
        return
    # Handle binary classification shap output and Explanation objects
    def _to_array(v):
        if hasattr(v, 'values'):
            return np.asarray(v.values)
        return np.asarray(v)
    if isinstance(shap_vals, list) and len(shap_vals) == 2:
        shap_arr = _to_array(shap_vals[1])
        # choose expected for positive class if available
        try:
            exp_arr = np.asarray(expected)
            expected_val = float(exp_arr[1] if exp_arr.size >= 2 else exp_arr.ravel()[0])
        except Exception:
            try:
                expected_val = float(expected)
            except Exception:
                expected_val = None
    else:
        shap_arr = _to_array(shap_vals)
        try:
            exp_arr = np.asarray(expected)
            expected_val = float(exp_arr.ravel()[0]) if exp_arr.size >= 1 else None
        except Exception:
            try:
                expected_val = float(expected)
            except Exception:
                expected_val = None
    if shap_arr is None or (isinstance(shap_arr, np.ndarray) and shap_arr.size == 0):
        return

    # Global importance: mean absolute SHAP
    mean_abs = np.mean(np.abs(np.asarray(shap_arr)), axis=0)
    mean_abs = np.asarray(mean_abs).ravel()
    # Ensure feature names length aligns with SHAP feature dimension
    n_features = int(mean_abs.shape[0])
    if len(feat_names) != n_features:
        feat_names = [f'f{i}' for i in range(n_features)]
    order = np.argsort(mean_abs)[::-1]
    top_n = min(20, n_features)
    top_idx = np.asarray(order[:top_n]).astype(int).ravel().tolist()
    glob_df = pd.DataFrame({
        'feature': [feat_names[int(i)] for i in top_idx],
        'mean_abs_shap': [float(mean_abs[int(i)]) for i in top_idx]
    })
    # Bar plot
    plt.figure(figsize=(8.0, 5.2))
    sns.barplot(data=glob_df, x='mean_abs_shap', y='feature', palette='viridis')
    plt.xlabel('Mean |SHAP|')
    plt.ylabel('Feature')
    plt.title('Global feature importance (SHAP) — top 20')
    _savefig(os.path.join(VIS, 'shap_global_bar.svg'))

    # Beeswarm (if available)
    try:
        shap.summary_plot(shap_arr, Xt, feature_names=feat_names, show=False, max_display=20)
        _savefig(os.path.join(VIS, 'shap_global_beeswarm.svg'))
    except Exception:
        pass

    # Per-case top contributions (limit cases for size)
    n_cases = min(50, shap_arr.shape[0])
    rows = []
    for i in range(n_cases):
        sv = np.asarray(shap_arr[i]).ravel()
        xv = np.asarray(Xt[i]).ravel()
        idxs = np.argsort(np.abs(sv)).astype(int)[::-1][:10]
        for j in idxs:
            j_int = int(j)
            rows.append({
                'case_index': int(i),
                'feature': feat_names[j_int] if j_int < len(feat_names) else f'f{j_int}',
                'shap_value': float(sv[j_int]),
                'abs_shap': float(abs(sv[j_int])),
                'feature_value': float(xv[j_int]) if j_int < len(xv) else None,
                'expected_value': expected_val if isinstance(expected_val, (int, float)) else None,
            })
    out_csv = os.path.join(REPORTS, 'phase4b_shap_per_case_top.csv')
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def gen_subgroup_fairness():
    # Load model and raw data
    model_path = os.path.join(BASE, 'models', 'phase4b_best_calibrated_model.joblib')
    meta_path = os.path.join(BASE, 'models', 'phase4b_best_model_meta.json')
    data_path = os.path.join(BASE, '..', 'clean_data.csv')
    if not (os.path.exists(model_path) and os.path.exists(meta_path) and os.path.exists(data_path)):
        return
    _ensure_custom_classes_for_unpickle()
    clf = joblib.load(model_path)
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    thr = float(meta.get('threshold', 0.5))
    df = pd.read_csv(data_path)
    # Detect target
    # Prefer columns containing "death" or "outcome"
    candidates = [c for c in df.columns if 'death' in c.lower()] or [c for c in df.columns if 'outcome' in c.lower()]
    if not candidates:
        return
    target = candidates[0]
    y = _to_binary_series(df[target])
    # Drop rows with nan target
    work = df.copy()
    work['__y__'] = y
    work = work.dropna(subset=['__y__'])
    y_bin = work['__y__'].astype(int)
    X = work[[c for c in work.columns if c != target and c != '__y__']]
    try:
        probs = clf.predict_proba(X)[:, 1]
    except Exception:
        return
    preds = (probs >= thr).astype(int)

    # Overall metrics for parity comparisons
    tp_all = int(((preds == 1) & (y_bin == 1)).sum())
    tn_all = int(((preds == 0) & (y_bin == 0)).sum())
    fp_all = int(((preds == 1) & (y_bin == 0)).sum())
    fn_all = int(((preds == 0) & (y_bin == 1)).sum())
    precision_all = tp_all / (tp_all + fp_all) if (tp_all + fp_all) > 0 else np.nan
    recall_all = tp_all / (tp_all + fn_all) if (tp_all + fn_all) > 0 else np.nan
    specificity_all = tn_all / (tn_all + fp_all) if (tn_all + fp_all) > 0 else np.nan
    f1_all = (2 * precision_all * recall_all) / (precision_all + recall_all) if precision_all and recall_all and (precision_all + recall_all) > 0 else np.nan
    pos_rate_all = float(preds.mean()) if len(preds) > 0 else np.nan

    PARITY_DELTA = 0.10  # tolerance for parity checks

    subgroup_rows = []
    def add_metric(label: str, mask: pd.Series):
        m = mask & (y_bin.isin([0,1]))
        if m.sum() == 0:
            return
        yt = y_bin[m]
        pdx = preds[m]
        pos = (yt == 1).sum()
        neg = (yt == 0).sum()
        tp = int(((pdx == 1) & (yt == 1)).sum())
        tn = int(((pdx == 0) & (yt == 0)).sum())
        fp = int(((pdx == 1) & (yt == 0)).sum())
        fn = int(((pdx == 0) & (yt == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        f1 = (2 * precision * recall) / (precision + recall) if precision and recall and (precision + recall) > 0 else np.nan
        pos_rate = pdx.mean() if len(pdx) > 0 else np.nan
        # Parity gaps and flags
        def gap(val, base):
            return float(val - base) if (val is not None and base is not None and not np.isnan(val) and not np.isnan(base)) else np.nan
        def fail(val, base):
            return bool((val is not None and base is not None and not np.isnan(val) and not np.isnan(base)) and (abs(val - base) > PARITY_DELTA))
        recall_gap = gap(recall, recall_all)
        precision_gap = gap(precision, precision_all)
        specificity_gap = gap(specificity, specificity_all)
        pos_rate_gap = gap(pos_rate, pos_rate_all)
        fail_recall = fail(recall, recall_all)
        fail_precision = fail(precision, precision_all)
        fail_specificity = fail(specificity, specificity_all)
        fail_pos_rate = fail(pos_rate, pos_rate_all)
        any_fail = fail_recall or fail_precision or fail_specificity or fail_pos_rate
        subgroup_rows.append({
            'subgroup': label,
            'n': int(m.sum()),
            'positives': int(pos),
            'negatives': int(neg),
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1': f1,
            'positive_rate': float(pos_rate) if not np.isnan(pos_rate) else np.nan,
            'recall_gap': recall_gap,
            'precision_gap': precision_gap,
            'specificity_gap': specificity_gap,
            'positive_rate_gap': pos_rate_gap,
            'fail_recall_parity': fail_recall,
            'fail_precision_parity': fail_precision,
            'fail_specificity_parity': fail_specificity,
            'fail_positive_rate_parity': fail_pos_rate,
            'any_parity_fail': any_fail,
        })

    # Gender
    if 'Gender' in work.columns:
        add_metric('Gender=Male', work['Gender'].astype(str).str.lower().str.contains('male'))
        add_metric('Gender=Female', work['Gender'].astype(str).str.lower().str.contains('female'))
    # Age bands
    if 'Age' in work.columns:
        age = pd.to_numeric(work['Age'], errors='coerce')
        add_metric('Age<50', age < 50)
        add_metric('Age50-64', (age >= 50) & (age <= 64))
        add_metric('Age65-74', (age >= 65) & (age <= 74))
        add_metric('Age>=75', age >= 75)
    # Comorbidities
    for col, name in [
        ('Known case of Hypertension (YES/NO)', 'Hypertension=YES'),
        ('known case of diabetes (YES/NO)', 'Diabetes=YES'),
        ('known case of atrial fibrillation (YES/NO)', 'AFib=YES'),
    ]:
        if col in work.columns:
            val = work[col].astype(str).str.strip().str.lower()
            add_metric(name, val.isin(['yes','y','1','true','t']))

    if not subgroup_rows:
        return
    sub_df = pd.DataFrame(subgroup_rows)
    out_csv = os.path.join(REPORTS, 'phase4b_subgroup_metrics.csv')
    sub_df.to_csv(out_csv, index=False)

    # Save parity flags JSON summary
    flags = []
    for r in sub_df.itertuples(index=False):
        flag_names = []
        if r.fail_recall_parity:
            flag_names.append('recall_parity')
        if r.fail_precision_parity:
            flag_names.append('precision_parity')
        if r.fail_specificity_parity:
            flag_names.append('specificity_parity')
        if r.fail_positive_rate_parity:
            flag_names.append('positive_rate_parity')
        if flag_names:
            flags.append({'subgroup': r.subgroup, 'flags': flag_names, 'n': int(r.n)})
    disparity_summary = {
        'overall': {
            'precision': precision_all,
            'recall': recall_all,
            'specificity': specificity_all,
            'f1': f1_all,
            'positive_rate': pos_rate_all,
        },
        'parity_delta': PARITY_DELTA,
        'num_flagged_subgroups': len(flags),
    }
    with open(os.path.join(REPORTS, 'phase4b_fairness_flags.json'), 'w') as f:
        json.dump({'flags': flags, 'disparity_summary': disparity_summary}, f, indent=2)

    # Visual: recall by subgroup (color-coded by parity fail)
    plot_df = sub_df.dropna(subset=['recall']).copy()
    if not plot_df.empty:
        plt.figure(figsize=(8.6, 5.0))
        colors = plot_df['any_parity_fail'].map({True: '#EF5350', False: '#66BB6A'}).tolist()
        sns.barplot(data=plot_df, x='subgroup', y='recall', palette=colors)
        plt.xticks(rotation=30, ha='right')
        plt.ylabel('Recall (sensitivity)')
        plt.title('Subgroup recall — fairness check (red = parity flag)')
        for i, r in enumerate(plot_df.itertuples(index=False)):
            if not np.isnan(r.recall):
                lbl = f"{r.recall:.2f}{'\nFAIL' if r.any_parity_fail else ''}"
                plt.text(i, r.recall + 0.02, lbl, ha='center', fontsize=9)
        _savefig(os.path.join(VIS, 'fairness_recall_by_subgroup.svg'))


def main():
    gen_augmentation_intensity()
    gen_hpo_scatter()
    gen_sampler_gate_scatter()
    # Layperson visuals
    gen_class_balance_pie()
    gen_augmentation_cap_bar()
    gen_model_leaderboard()
    gen_threshold_explainer()
    # New: decision curve and subgroup fairness
    gen_decision_curve()
    gen_subgroup_fairness()
    # New: SHAP explanations
    gen_shap_explanations()
    print('New visuals generated in phase4b/visuals:')
    print(' - augmentation_intensity.svg')
    print(' - hpo_scatter_smote_nearmiss.svg (if data available)')
    print(' - hpo_scatter_adasyn_nearmiss.svg (if data available)')
    print(' - sampler_overfitting_gate_f1.svg')
    print(' - sampler_overfitting_gate_auc.svg')
    print(' - layperson_class_balance.svg (if raw data available)')
    print(' - layperson_augmentation_cap.svg (uses best params caps or defaults)')
    print(' - layperson_model_leaderboard.svg')
    print(' - layperson_threshold_explainer.svg (if sweep available)')
    print(' - decision_curve.svg (if prevalence & sweep available)')
    print(' - fairness_recall_by_subgroup.svg (if model & data available)')
    print(' - shap_global_bar.svg (if model & data available)')
    print(' - shap_global_beeswarm.svg (if model & data available)')


def gen_augmentation_intensity():
    # Collect sampling_strategy explored for SMOTE+NearMiss and ADASYN+NearMiss
    smn = load_csv('smote_nearmiss_hpo_cv_results.csv')
    adn = load_csv('adasyn_nearmiss_hpo_cv_results.csv')
    rows = []
    if not smn.empty:
        col = 'param_sampler_smote__sampling_strategy'
        if col in smn.columns:
            for v in smn[col].dropna().values:
                rows.append({'sampler': 'SMOTE+NearMiss', 'sampling_strategy': v})
    if not adn.empty:
        col = 'param_sampler_adasyn__sampling_strategy'
        if col in adn.columns:
            for v in adn[col].dropna().values:
                rows.append({'sampler': 'ADASYN+NearMiss', 'sampling_strategy': v})
    if not rows:
        return
    df = pd.DataFrame(rows)

    plt.figure(figsize=(7, 4))
    sns.boxplot(data=df, x='sampler', y='sampling_strategy', palette='Set2')
    sns.stripplot(data=df, x='sampler', y='sampling_strategy', color='k', alpha=0.5, jitter=0.15)
    plt.title('Augmentation intensity explored (sampling_strategy)\nNote: 1.0 equals fully balancing minority to majority count')
    plt.ylabel('sampling_strategy (higher = more synthetic minority)')
    _savefig(os.path.join(VIS, 'augmentation_intensity.svg'))


if __name__ == '__main__':
    main()