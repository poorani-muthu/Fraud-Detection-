"""
FraudGuard Analysis Engine
Dataset: Kaggle Credit Card Fraud (V1-V28 PCA features, Amount, Time)
Models:  Logistic Regression, Random Forest, Gradient Boosting
Extras:  SMOTE, threshold analysis, feature importance, SHAP-style explanations
"""
import json, os, math, warnings, pickle, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                              precision_score, recall_score, accuracy_score,
                              confusion_matrix, roc_curve, precision_recall_curve)
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, 'analysis'))
from smote import smote_oversample

def clean(obj):
    if isinstance(obj, float):
        return 0.0 if (math.isnan(obj) or math.isinf(obj)) else round(obj, 6)
    if isinstance(obj, (np.floating,)): return clean(float(obj))
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, dict):  return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray): return clean(obj.tolist())
    return obj

def run():
    print("="*56)
    print("  FraudGuard — Credit Card Fraud Detection Pipeline")
    print("="*56)

    # ── 1. LOAD ────────────────────────────────────────────────
    print("\n[1/7] Loading Kaggle Credit Card Fraud dataset...")
    df = pd.read_csv(os.path.join(BASE, 'data', 'creditcard.csv'))
    n_fraud = int(df.Class.sum())
    n_total = len(df)
    fraud_pct = df.Class.mean() * 100
    print(f"  {n_total:,} transactions | {n_fraud} fraud ({fraud_pct:.3f}%)")

    # Feature sets
    V_COLS    = [f'V{i}' for i in range(1, 29)]
    FEATURES  = V_COLS + ['Amount', 'Time']
    FEAT_DISP = V_COLS + ['Amount', 'Time']

    # Scale Amount and Time (V features already PCA-scaled)
    df['Amount_sc'] = StandardScaler().fit_transform(df[['Amount']])
    df['Time_sc']   = StandardScaler().fit_transform(df[['Time']])
    FEATURES_SC = V_COLS + ['Amount_sc', 'Time_sc']

    X = df[FEATURES_SC].values.astype(float)
    y = df['Class'].values.astype(int)
    X_raw = df[FEATURES].values  # for display

    X_train, X_test, y_train, y_test, Xr_train, Xr_test = train_test_split(
        X, y, X_raw, test_size=0.2, stratify=y, random_state=42)
    print(f"  Train {len(X_train):,} | Test {len(X_test):,} | Test fraud: {y_test.sum()}")

    # ── 2. EDA ─────────────────────────────────────────────────
    print("\n[2/7] EDA statistics...")
    fdf = df[df.Class == 1]
    ldf = df[df.Class == 0]

    # Amount distributions
    bins  = [0, 1, 5, 10, 50, 100, 500, 1000, 5000, 1e9]
    lbls  = ['<1','1-5','5-10','10-50','50-100','100-500','500-1k','1k-5k','5k+']
    amt_fraud = [int(((fdf.Amount>=bins[i])&(fdf.Amount<bins[i+1])).sum()) for i in range(len(lbls))]
    amt_legit = [int(((ldf.Amount>=bins[i])&(ldf.Amount<bins[i+1])).sum()) for i in range(len(lbls))]

    # Time distribution (hours)
    fdf_h = (fdf.Time / 3600).astype(int) % 24
    ldf_h = (ldf.Time / 3600).astype(int) % 24
    hour_fraud = [int((fdf_h == h).sum()) for h in range(24)]
    hour_legit_norm = [int(round((ldf_h == h).sum() * n_fraud / len(ldf))) for h in range(24)]

    # V-feature means comparison (top 10 most different)
    v_diffs = {}
    for v in V_COLS:
        v_diffs[v] = abs(float(fdf[v].mean()) - float(ldf[v].mean()))
    top_v = sorted(v_diffs, key=v_diffs.get, reverse=True)[:10]

    # Correlations with Class
    corr_cols = V_COLS[:14] + ['Amount', 'Time']
    corrs = df[corr_cols + ['Class']].corr()['Class'].drop('Class').sort_values(key=abs, ascending=False)

    eda = {
        'dataset': {
            'total': n_total, 'fraud': n_fraud, 'legit': n_total - n_fraud,
            'fraud_rate': float(fraud_pct), 'n_features': len(FEATURES),
            'source': 'Kaggle Credit Card Fraud Detection (ULB)',
        },
        'amount_dist': {
            'bins': lbls, 'fraud': amt_fraud, 'legit': amt_legit,
            'fraud_mean': float(fdf.Amount.mean()),
            'legit_mean': float(ldf.Amount.mean()),
        },
        'hour_dist': {
            'hours': list(range(24)),
            'fraud': hour_fraud,
            'legit_norm': hour_legit_norm,
        },
        'v_comparison': {
            'features': top_v,
            'fraud_means': [float(fdf[v].mean()) for v in top_v],
            'legit_means': [float(ldf[v].mean()) for v in top_v],
        },
        'correlations': {
            'features': list(corrs.index[:12]),
            'values':   [float(v) for v in corrs.values[:12]],
        },
        'feature_stats': {v: {
            'fraud_mean': float(fdf[v].mean()),
            'legit_mean': float(ldf[v].mean()),
            'fraud_std':  float(fdf[v].std()),
            'legit_std':  float(ldf[v].std()),
        } for v in top_v},
    }

    # ── 3. SMOTE ────────────────────────────────────────────────
    print("\n[3/7] SMOTE oversampling...")
    X_res, y_res = smote_oversample(X_train, y_train, target_ratio=0.2, k=5, random_state=42)
    print(f"  Before: {y_train.sum()} fraud / {(y_train==0).sum():,} legit")
    print(f"  After:  {int(y_res.sum())} fraud / {int((y_res==0).sum()):,} legit")
    smote_info = {
        'before_fraud': int(y_train.sum()), 'before_legit': int((y_train==0).sum()),
        'after_fraud':  int(y_res.sum()),   'after_legit':  int((y_res==0).sum()),
    }

    # ── 4. TRAIN ────────────────────────────────────────────────
    print("\n[4/7] Training models...")
    mdefs = {
        'Logistic Regression': LogisticRegression(C=0.01, max_iter=1000, random_state=42, class_weight='balanced'),
        'Random Forest':       RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=5, random_state=42, n_jobs=-1),
        'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42),
    }
    results, trained = {}, {}
    best_name, best_auc = None, 0

    for name, model in mdefs.items():
        print(f"  {name}...", end=' ', flush=True)
        model.fit(X_res, y_res)
        proba = model.predict_proba(X_test)[:, 1]
        pred  = (proba >= 0.5).astype(int)

        auc = roc_auc_score(y_test, proba)
        ap  = average_precision_score(y_test, proba)
        f1  = f1_score(y_test, pred, zero_division=0)
        pr  = precision_score(y_test, pred, zero_division=0)
        re  = recall_score(y_test, pred, zero_division=0)
        acc = accuracy_score(y_test, pred)
        cm  = confusion_matrix(y_test, pred)

        fpr, tpr, _ = roc_curve(y_test, proba)
        pc, rc, _   = precision_recall_curve(y_test, proba)
        s1 = max(1, len(fpr)//80); s2 = max(1, len(pc)//80)

        thr_rows = []
        for t in np.arange(0.1, 1.0, 0.1):
            p = (proba >= t).astype(int)
            thr_rows.append({
                'threshold': round(float(t), 1),
                'precision': float(precision_score(y_test, p, zero_division=0)),
                'recall':    float(recall_score(y_test, p, zero_division=0)),
                'f1':        float(f1_score(y_test, p, zero_division=0)),
                'fp': int(((p==1)&(y_test==0)).sum()),
                'fn': int(((p==0)&(y_test==1)).sum()),
            })

        results[name] = {
            'auc': float(auc), 'ap': float(ap), 'f1': float(f1),
            'precision': float(pr), 'recall': float(re), 'accuracy': float(acc),
            'tp': int(cm[1,1]), 'fp': int(cm[0,1]),
            'fn': int(cm[1,0]), 'tn': int(cm[0,0]),
            'roc': {'fpr':[float(v) for v in fpr[::s1]], 'tpr':[float(v) for v in tpr[::s1]]},
            'prc': {'precision':[float(v) for v in pc[::s2]], 'recall':[float(v) for v in rc[::s2]]},
            'thresholds': thr_rows,
        }
        trained[name] = model
        print(f"AUC={auc:.4f}  F1={f1:.4f}  Recall={re:.4f}")
        if auc > best_auc: best_auc = auc; best_name = name

    print(f"\n  Best: {best_name}  AUC={best_auc:.4f}")

    # ── 5. FEATURE IMPORTANCE ───────────────────────────────────
    print("\n[5/7] Feature importance...")
    rf_imp = trained['Random Forest'].feature_importances_
    gb_imp = trained['Gradient Boosting'].feature_importances_
    lr_imp = np.abs(trained['Logistic Regression'].coef_[0])
    lr_imp = lr_imp / (lr_imp.sum() + 1e-8)

    importance = {k: {'features': FEAT_DISP, 'values': [float(v) for v in imp]}
                  for k, imp in [('Random Forest',rf_imp),('Gradient Boosting',gb_imp),('Logistic Regression',lr_imp)]}

    # ── 6. SCORE DIST + SHAP-STYLE ──────────────────────────────
    print("\n[6/7] Score distributions & explanations...")
    best_model = trained[best_name]
    all_proba  = best_model.predict_proba(X_test)[:, 1]
    best_imp   = rf_imp if best_name == 'Random Forest' else gb_imp

    sbins = np.arange(0, 1.05, 0.05)
    score_dist = {
        'bins':  [round(float(b), 2) for b in sbins[:-1]],
        'fraud': [int(((all_proba[y_test==1]>=sbins[i])&(all_proba[y_test==1]<sbins[i+1])).sum()) for i in range(len(sbins)-1)],
        'legit': [int(((all_proba[y_test==0]>=sbins[i])&(all_proba[y_test==0]<sbins[i+1])).sum()) for i in range(len(sbins)-1)],
    }

    fi = np.where(y_test == 1)[0]
    n_top = min(5, len(fi))
    top_idx = fi[np.argsort(all_proba[fi])[-n_top:]]
    mn = X_train.mean(axis=0); sd = X_train.std(axis=0) + 1e-8
    explanations = []
    for idx in top_idx:
        row = X_test[idx]
        contrib = ((row - mn) / sd) * best_imp
        mx = np.abs(contrib).max() + 1e-8
        explanations.append({
            'prob':          float(all_proba[idx]),
            'features':      FEAT_DISP,
            'contributions': [float(v/mx) for v in contrib],
            'raw_values':    [float(Xr_test[idx][i]) for i in range(len(FEATURES))],
        })

    # ── 7. SAVE ─────────────────────────────────────────────────
    print("\n[7/7] Saving...")
    pkl_path = os.path.join(BASE, 'models', 'best_model.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump({
            'model': best_model, 'name': best_name,
            'features': FEATURES_SC,
            'feat_display': FEAT_DISP,
            'amount_scaler_mean': float(df['Amount'].mean()),
            'amount_scaler_std':  float(df['Amount'].std()),
            'time_scaler_mean':   float(df['Time'].mean()),
            'time_scaler_std':    float(df['Time'].std()),
        }, f)

    analysis = {
        'meta': {'best_model': best_name, 'best_auc': float(best_auc), 'models': list(results.keys())},
        'eda': eda, 'smote': smote_info,
        'models': results, 'importance': importance,
        'score_dist': score_dist, 'explanations': explanations,
    }
    json_path = os.path.join(BASE, 'static', 'analysis.json')
    with open(json_path, 'w') as f:
        json.dump(clean(analysis), f, separators=(',', ':'))

    print(f"  analysis.json: {os.path.getsize(json_path)//1024}KB")
    print(f"  best_model.pkl saved")
    print(f"\n  DONE — {best_name}  AUC={best_auc:.4f}")
    return analysis

if __name__ == '__main__':
    run()
