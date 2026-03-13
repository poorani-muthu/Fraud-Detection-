"""
Generates data matching Kaggle Credit Card Fraud Detection dataset.
Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Features: Time, V1-V28 (PCA anonymised), Amount, Class
"""
import numpy as np
import pandas as pd
import os

np.random.seed(42)
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_LEGIT = 49902
N_FRAUD = 98

def gen_legit(n):
    V = np.random.randn(n, 28)
    V[:,0] += -0.7; V[:,2] += -0.4; V[:,4] += 0.3; V[:,6] += 0.2
    df = pd.DataFrame(V, columns=[f'V{i+1}' for i in range(28)])
    df['Time']   = np.random.uniform(0, 172792, n)
    df['Amount'] = np.random.lognormal(3.0, 1.5, n).clip(0, 25691)
    df['Class']  = 0
    return df

def gen_fraud(n):
    V = np.random.randn(n, 28) * 1.3
    V[:,13] += -8.0
    V[:,3]  +=  4.0
    V[:,10] +=  4.0
    V[:,2]  += -5.0
    V[:,16] += -3.0
    V[:,11] +=  3.5
    V[:,1]  += -3.5
    V[:,6]  += -2.5
    df = pd.DataFrame(V, columns=[f'V{i+1}' for i in range(28)])
    df['Time']   = np.random.uniform(0, 172792, n)
    df['Amount'] = np.random.lognormal(2.5, 2.0, n).clip(0, 2126)
    df['Class']  = 1
    return df

if __name__ == '__main__':
    os.makedirs(os.path.join(BASE, 'data'), exist_ok=True)
    df = pd.concat([gen_legit(N_LEGIT), gen_fraud(N_FRAUD)], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    out = os.path.join(BASE, 'data', 'creditcard.csv')
    df.to_csv(out, index=False)
    print(f"Dataset: {len(df):,} rows | {df.Class.sum()} fraud ({df.Class.mean()*100:.3f}%)")
