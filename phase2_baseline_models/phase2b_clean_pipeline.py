#!/usr/bin/env python3
import os, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.base import clone

DATA_PATH = Path(__file__).resolve().parent.parent / 'clean_data.csv'
TEST_CASES_PATH = Path(__file__).resolve().parent.parent / 'phase3_ensemble_methods/visuals/test_cases/comprehensive_test_case_parameters.json'
OUT_DIR = Path(__file__).resolve().parent
REPORTS_DIR = OUT_DIR / 'reports'
VIS_DIR = OUT_DIR / 'visuals'
REPORTS_DIR.mkdir(exist_ok=True, parents=True)
VIS_DIR.mkdir(exist_ok=True, parents=True)

class Phase2BClean:
    def __init__(self, data_path=DATA_PATH):
        self.data_path = Path(data_path)
        self.df = None
        self.feature_cols = None
        self.cat_cols = []
        self.num_cols = []
        self.le_map = {}
        self.imputer_num = SimpleImputer(strategy='median')
        self.imputer_cat = SimpleImputer(strategy='most_frequent')
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.fitted_models = {}

    def load_data(self):
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data not found: {self.data_path}")
        df = pd.read_csv(self.data_path)
        if 'Death outcome (YES/NO)' not in df.columns:
            raise ValueError("Column 'Death outcome (YES/NO)' missing")
        df['mortality'] = df['Death outcome (YES/NO)'].astype(str).str.lower().map({'yes':1,'no':0})
        # Drop rows with unknown/NaN mortality
        df = df[~df['mortality'].isna()].copy()
        self.df = df

    def preprocess(self):
        df = self.df.copy()
        y = df['mortality'].values
        drop_cols = [c for c in df.columns if c == 'mortality' or c.lower().startswith('death outcome')]
        X = df.drop(columns=drop_cols)
        self.cat_cols = [c for c in X.columns if X[c].dtype=='object']
        self.num_cols = [c for c in X.columns if c not in self.cat_cols]
        X_num = pd.DataFrame(self.imputer_num.fit_transform(X[self.num_cols]), columns=self.num_cols)
        # Normalize categoricals
        X_cat_raw = X[self.cat_cols].astype(str).apply(lambda col: col.str.strip().str.lower())
        X_cat_imp = pd.DataFrame(self.imputer_cat.fit_transform(X_cat_raw), columns=self.cat_cols)
        for c in self.cat_cols:
            le = LabelEncoder()
            X_cat_imp[c] = le.fit_transform(X_cat_imp[c].astype(str))
            self.le_map[c] = le
        X_all = pd.concat([X_num, X_cat_imp], axis=1)
        X_scaled = self.scaler.fit_transform(X_all)
        self.feature_cols = list(X_all.columns)
        return X_scaled, y

    def init_models(self):
        self.models = {
            'rf_standard': RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42),
            'rf_balanced': RandomForestClassifier(n_estimators=200, max_depth=12, class_weight='balanced', random_state=42),
            'lr_standard': LogisticRegression(max_iter=2000, random_state=42),
            'lr_balanced': LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
        }

    def train_and_eval(self, X_train, X_test, y_train, y_test, tag):
        for name, model in self.models.items():
            mname = f"{name}_{tag}"
            fitted = clone(model)
            fitted.fit(X_train, y_train)
            y_pred = fitted.predict(X_test)
            y_prob = fitted.predict_proba(X_test)[:,1] if hasattr(fitted,'predict_proba') else y_pred
            self.results[mname] = {
                'tag': tag,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='binary', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='binary', zero_division=0),
                'f1': f1_score(y_test, y_pred, average='binary', zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_prob) if len(np.unique(y_test))==2 else np.nan
            }
            self.fitted_models[mname] = fitted
        return True

    def save_metrics_table(self, fname_csv):
        df = pd.DataFrame(self.results).T.reset_index().rename(columns={'index':'model'})
        order = ['model','tag','accuracy','precision','recall','f1','roc_auc']
        df = df[order]
        df.sort_values(['f1','roc_auc','accuracy'], ascending=False, inplace=True)
        df.to_csv(REPORTS_DIR / fname_csv, index=False)
        with open(REPORTS_DIR / (Path(fname_csv).stem + '.json'), 'w') as f:
            json.dump(df.to_dict(orient='records'), f, indent=2)
        return df

    def _align_and_transform_cases(self, cases_df):
        # Ensure all training features exist
        df = cases_df.copy()
        # Keep only known columns
        df = df[[c for c in df.columns if c in self.feature_cols]]
        # Add missing training columns
        for c in self.feature_cols:
            if c not in df.columns:
                if c in self.num_cols:
                    df[c] = self.imputer_num.statistics_[self.num_cols.index(c)] if hasattr(self.imputer_num,'statistics_') else 0
                else:
                    df[c] = self.imputer_cat.statistics_[self.cat_cols.index(c)] if hasattr(self.imputer_cat,'statistics_') else 'unknown'
        # Reorder
        df = df[self.feature_cols]
        # Split back to num/cat
        # Impute numeric (safety)
        if self.num_cols:
            df[self.num_cols] = self.imputer_num.transform(df[self.num_cols])
        # Normalize and encode categoricals safely
        if self.cat_cols:
            for c in self.cat_cols:
                vals = df[c].astype(str).str.strip().str.lower()
                if c in self.le_map:
                    le = self.le_map[c]
                    mapping = {cls:i for i, cls in enumerate(le.classes_)}
                    df[c] = vals.map(lambda v: mapping.get(v, 0)).astype(int)
                else:
                    df[c] = 0
        X_scaled = self.scaler.transform(df)
        return X_scaled

    @staticmethod
    def prob_to_band(p):
        return 'Low' if p < 0.33 else ('Moderate' if p < 0.66 else 'High')

    def evaluate_on_test_cases(self, top_models):
        with open(TEST_CASES_PATH,'r') as f:
            tc = json.load(f)
        rows = []
        cases = []
        for case in tc['test_cases']:
            params = case['parameters'].copy()
            expected = params.pop('expected_risk', None)
            cases.append((case['case_id'], case['name'], expected, params))
        cases_df = pd.DataFrame([p for (_,_,_,p) in cases])
        X_cases = self._align_and_transform_cases(cases_df)
        summary = {}
        for mname in top_models:
            model = self.fitted_models.get(mname)
            if model is None:
                continue
            prob = model.predict_proba(X_cases)[:,1] if hasattr(model,'predict_proba') else model.predict(X_cases)
            pred = (prob>=0.5).astype(int)
            pred_band = [self.prob_to_band(p) for p in prob]
            correct = [int(pb==exp) for (_,_,exp,_), pb in zip(cases, pred_band)]
            for (cid, cname, exp, _), p, y, pb, ok in zip(cases, prob, pred, pred_band, correct):
                rows.append({'model': mname, 'case_id': cid, 'case_name': cname, 'expected_risk': exp, 'pred_prob': float(p), 'pred_label': int(y), 'pred_risk_band': pb, 'correct_band': int(ok)})
            summary[mname] = {'test_case_correct': int(np.sum(correct)), 'test_case_total': len(correct), 'test_case_accuracy': float(np.mean(correct))}
        res_df = pd.DataFrame(rows)
        res_df.to_csv(REPORTS_DIR / 'phase2b_test_cases_results.csv', index=False)
        with open(REPORTS_DIR / 'phase2b_test_cases_summary.json','w') as f:
            json.dump(summary, f, indent=2)
        return res_df, summary

    def run(self):
        self.load_data()
        X, y = self.preprocess()
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        # Initial models
        self.init_models()
        self.train_and_eval(Xtr, Xte, ytr, yte, tag='nosmote')
        # SMOTE augmentation
        sm = SMOTE(random_state=42)
        Xtr_s, ytr_s = sm.fit_resample(Xtr, ytr)
        self.train_and_eval(Xtr_s, Xte, ytr_s, yte, tag='smote')
        # Save metrics table
        metrics_df = self.save_metrics_table('phase2b_metrics.csv')
        # Pick top models by F1
        top = metrics_df.sort_values(['f1','roc_auc','accuracy'], ascending=False).head(3)['model'].tolist()
        # Evaluate against 10 test cases
        cases_df, summary = self.evaluate_on_test_cases(top)
        print(f"Saved metrics to {REPORTS_DIR/'phase2b_metrics.csv'}")
        print(f"Saved test-case results to {REPORTS_DIR/'phase2b_test_cases_results.csv'}")
        print("Top by F1:", metrics_df.head(3)['model'].tolist())
        return True

if __name__ == '__main__':
    pipeline = Phase2BClean()
    ok = pipeline.run()
    exit(0 if ok else 1)