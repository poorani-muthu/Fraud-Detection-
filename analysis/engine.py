"""
FraudGuard Analysis Engine
Models: Logistic Regression, Random Forest, Gradient Boosting
Handles: SMOTE, class imbalance, feature importance, threshold analysis
"""
import json, os, math, warnings, pickle, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
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
    print("="*55)
    print("  FraudGuard — End-to-End Fraud Detection Pipeline")
    print("="*55)

    # ── 1. LOAD ───────────────────────────────────────────────
    print("\n[1/7] Loading data...")
    df = pd.read_csv(os.path.join(BASE, 'data', 'transactions.csv'))
    print(f"  {len(df):,} transactions | {df.is_fraud.sum():,} fraud ({df.is_fraud.mean()*100:.2f}%)")

    le = LabelEncoder()
    df['merchant_cat_enc'] = le.fit_transform(df['merchant_category'])

    FEATURES = ['amount','hour','day_of_week','merchant_cat_enc',
                'distance_from_home','distance_from_last','ratio_to_median',
                'pin_used','online_order','repeat_retailer','account_age_days',
                'transactions_24h','transactions_7d','v1','v2','v3','v4','v5']
    FEAT_DISPLAY = ['Amount','Hour','Day of Week','Merchant Cat',
                    'Dist. Home','Dist. Last Txn','Ratio to Median',
                    'PIN Used','Online Order','Repeat Retailer','Account Age (days)',
                    'Txns 24h','Txns 7d','V1','V2','V3','V4','V5']

    X = df[FEATURES].values.astype(float)
    y = df['is_fraud'].values.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    print(f"  Train {len(X_train):,} | Test {len(X_test):,}")

    # ── 2. EDA ────────────────────────────────────────────────
    print("\n[2/7] EDA stats...")
    fdf = df[df.is_fraud==1]
    ldf = df[df.is_fraud==0]

    bins  = [0,10,50,100,200,500,1000,2000,5000,999999]
    lbls  = ['<10','10-50','50-100','100-200','200-500','500-1k','1k-2k','2k-5k','5k+']
    cats  = sorted(df.merchant_category.unique())

    eda = {
        'dataset': {
            'total': int(len(df)), 'fraud': int(df.is_fraud.sum()),
            'legit': int((df.is_fraud==0).sum()),
            'fraud_rate': float(df.is_fraud.mean()*100),
            'n_features': len(FEATURES),
        },
        'amount_dist': {
            'bins': lbls,
            'fraud': [int(((fdf.amount>=bins[i])&(fdf.amount<bins[i+1])).sum()) for i in range(len(lbls))],
            'legit': [int(((ldf.amount>=bins[i])&(ldf.amount<bins[i+1])).sum()) for i in range(len(lbls))],
            'fraud_mean': float(fdf.amount.mean()),
            'legit_mean': float(ldf.amount.mean()),
        },
        'hour_dist': {
            'hours': list(range(24)),
            'fraud': [int((fdf.hour==h).sum()) for h in range(24)],
            'legit_norm': [int(round((ldf.hour==h).sum()*len(fdf)/len(ldf))) for h in range(24)],
        },
        'merchant_dist': {
            'categories': cats,
            'fraud': [int((fdf.merchant_category==c).sum()) for c in cats],
            'legit': [int((ldf.merchant_category==c).sum()) for c in cats],
            'fraud_rate': [round(float(df[df.merchant_category==c].is_fraud.mean()*100),2) for c in cats],
        },
        'feature_comparison': {f: {
            'fraud_mean':   float(fdf[f].mean()),
            'legit_mean':   float(ldf[f].mean()),
            'fraud_median': float(fdf[f].median()),
            'legit_median': float(ldf[f].median()),
        } for f in ['amount','distance_from_home','distance_from_last',
                    'ratio_to_median','account_age_days','transactions_24h','transactions_7d']},
    }

    # correlations
    tmp = df[FEATURES + ['is_fraud']].copy()
    corrs = tmp.corr()['is_fraud'].drop('is_fraud').sort_values(key=abs, ascending=False)
    eda['correlations'] = {
        'features': [FEAT_DISPLAY[FEATURES.index(f)] if f in FEATURES else f for f in corrs.index[:12]],
        'values':   [float(v) for v in corrs.values[:12]],
    }

    # ── 3. SMOTE ──────────────────────────────────────────────
    print("\n[3/7] SMOTE...")
    scaler = StandardScaler()
    Xtr_sc = scaler.fit_transform(X_train)
    Xte_sc = scaler.transform(X_test)
    X_res, y_res = smote_oversample(Xtr_sc, y_train, target_ratio=0.3, k=5, random_state=42)
    print(f"  Before: {y_train.sum():,} fraud / {(y_train==0).sum():,} legit")
    print(f"  After:  {int(y_res.sum()):,} fraud / {int((y_res==0).sum()):,} legit")
    smote_info = {
        'before_fraud': int(y_train.sum()), 'before_legit': int((y_train==0).sum()),
        'after_fraud':  int(y_res.sum()),   'after_legit':  int((y_res==0).sum()),
    }

    # ── 4. TRAIN ──────────────────────────────────────────────
    print("\n[4/7] Training models...")
    mdefs = {
        'Logistic Regression': LogisticRegression(C=0.1, max_iter=1000, random_state=42, class_weight='balanced'),
        'Random Forest':       RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1, class_weight='balanced'),
        'Gradient Boosting':   GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=5, random_state=42),
    }
    results, trained = {}, {}
    best_name, best_auc = None, 0

    for name, model in mdefs.items():
        print(f"  {name}...", end=' ', flush=True)
        model.fit(X_res, y_res)
        proba = model.predict_proba(Xte_sc)[:,1]
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
        s1 = max(1, len(fpr)//80)
        s2 = max(1, len(pc)//80)

        thr_rows = []
        for t in np.arange(0.1, 1.0, 0.1):
            p = (proba >= t).astype(int)
            thr_rows.append({
                'threshold': round(float(t),1),
                'precision': float(precision_score(y_test,p,zero_division=0)),
                'recall':    float(recall_score(y_test,p,zero_division=0)),
                'f1':        float(f1_score(y_test,p,zero_division=0)),
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
        if auc > best_auc: best_auc=auc; best_name=name

    print(f"\n  Best: {best_name}  AUC={best_auc:.4f}")

    # ── 5. FEATURE IMPORTANCE ─────────────────────────────────
    print("\n[5/7] Feature importance...")
    rf_imp = trained['Random Forest'].feature_importances_
    gb_imp = trained['Gradient Boosting'].feature_importances_
    lr_imp = np.abs(trained['Logistic Regression'].coef_[0])
    lr_imp = lr_imp / lr_imp.sum()

    importance = {k: {
        'features': FEAT_DISPLAY,
        'values':   [float(v) for v in imp]
    } for k, imp in [('Random Forest',rf_imp),('Gradient Boosting',gb_imp),('Logistic Regression',lr_imp)]}

    # ── 6. SCORE DIST + EXPLANATIONS ─────────────────────────
    print("\n[6/7] Explanations...")
    best_model = trained[best_name]
    all_proba  = best_model.predict_proba(Xte_sc)[:,1]
    best_imp   = rf_imp if best_name=='Random Forest' else gb_imp

    sbins = np.arange(0,1.05,0.05)
    score_dist = {
        'bins':  [round(float(b),2) for b in sbins[:-1]],
        'fraud': [int(((all_proba[y_test==1]>=sbins[i])&(all_proba[y_test==1]<sbins[i+1])).sum()) for i in range(len(sbins)-1)],
        'legit': [int(((all_proba[y_test==0]>=sbins[i])&(all_proba[y_test==0]<sbins[i+1])).sum()) for i in range(len(sbins)-1)],
    }

    fi = np.where(y_test==1)[0]
    top5 = fi[np.argsort(all_proba[fi])[-5:]]
    mn = Xtr_sc.mean(axis=0); sd = Xtr_sc.std(axis=0)+1e-8
    explanations = []
    for idx in top5:
        row = Xte_sc[idx]
        contrib = ((row - mn)/sd) * best_imp
        mx = np.abs(contrib).max()+1e-8
        explanations.append({
            'prob':          float(all_proba[idx]),
            'features':      FEAT_DISPLAY,
            'contributions': [float(v/mx) for v in contrib],
            'raw_values':    [float(X_test[idx][i]) for i in range(len(FEATURES))],
        })

    # ── 7. SAVE ───────────────────────────────────────────────
    print("\n[7/7] Saving...")
    pkl_path = os.path.join(BASE, 'models', 'best_model.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump({'model': best_model, 'scaler': scaler,
                     'features': FEATURES, 'le': le, 'name': best_name}, f)

    analysis = {
        'meta':         {'best_model': best_name, 'best_auc': float(best_auc), 'models': list(results.keys())},
        'eda':          eda,
        'smote':        smote_info,
        'models':       results,
        'importance':   importance,
        'score_dist':   score_dist,
        'explanations': explanations,
    }
    json_path = os.path.join(BASE, 'static', 'analysis.json')
    with open(json_path, 'w') as f:
        json.dump(clean(analysis), f, separators=(',',':'))

    print(f"  analysis.json: {os.path.getsize(json_path)//1024}KB")
    print(f"  best_model.pkl saved")
    print(f"\n  DONE — {best_name}  AUC={best_auc:.4f}")
    return analysis

if __name__ == '__main__':
    run()
