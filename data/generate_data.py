import numpy as np
import pandas as pd

np.random.seed(42)
N_LEGIT = 49000
N_FRAUD = 1000

hr_legit = np.array([1,1,1,1,1,2,4,6,7,7,7,7,6,6,6,6,6,5,5,5,4,4,3,2], dtype=float)
hr_legit /= hr_legit.sum()
hr_fraud = np.array([6,7,8,8,7,6,5,4,4,4,4,4,4,4,4,4,4,4,4,4,3,3,4,5], dtype=float)
hr_fraud /= hr_fraud.sum()
mc = ['grocery','gas','restaurant','online','retail','travel','entertainment']

def gen_legit(n):
    return pd.DataFrame({
        'amount':            np.random.lognormal(3.5, 1.2, n).clip(0.5, 5000),
        'hour':              np.random.choice(24, n, p=hr_legit),
        'day_of_week':       np.random.randint(0, 7, n),
        'merchant_category': np.random.choice(mc, n, p=[0.25,0.15,0.18,0.20,0.12,0.05,0.05]),
        'distance_from_home':np.random.exponential(15, n).clip(0, 500),
        'distance_from_last':np.random.exponential(10, n).clip(0, 200),
        'ratio_to_median':   np.random.lognormal(0, 0.4, n).clip(0.1, 10),
        'pin_used':          np.random.binomial(1, 0.7, n),
        'online_order':      np.random.binomial(1, 0.2, n),
        'repeat_retailer':   np.random.binomial(1, 0.8, n),
        'account_age_days':  np.random.gamma(5, 200, n).clip(1, 3650),
        'transactions_24h':  np.random.poisson(3, n).clip(0, 20),
        'transactions_7d':   np.random.poisson(15, n).clip(0, 100),
        'v1': np.random.normal(0, 1, n),
        'v2': np.random.normal(0, 1, n),
        'v3': np.random.normal(0, 1, n),
        'v4': np.random.normal(0, 1, n),
        'v5': np.random.normal(0, 1, n),
        'is_fraud': 0
    })

def gen_fraud(n):
    return pd.DataFrame({
        'amount':            np.random.lognormal(4.5, 1.5, n).clip(10, 8000),
        'hour':              np.random.choice(24, n, p=hr_fraud),
        'day_of_week':       np.random.randint(0, 7, n),
        'merchant_category': np.random.choice(mc, n, p=[0.05,0.05,0.05,0.45,0.15,0.15,0.10]),
        'distance_from_home':np.random.exponential(80, n).clip(0, 2000),
        'distance_from_last':np.random.exponential(150, n).clip(0, 3000),
        'ratio_to_median':   np.random.lognormal(1.5, 0.8, n).clip(0.5, 50),
        'pin_used':          np.random.binomial(1, 0.2, n),
        'online_order':      np.random.binomial(1, 0.7, n),
        'repeat_retailer':   np.random.binomial(1, 0.2, n),
        'account_age_days':  np.random.gamma(1, 50, n).clip(1, 500),
        'transactions_24h':  np.random.poisson(8, n).clip(0, 30),
        'transactions_7d':   np.random.poisson(25, n).clip(0, 120),
        'v1': np.random.normal(-2, 1.5, n),
        'v2': np.random.normal(2, 1.5, n),
        'v3': np.random.normal(-1.5, 1.5, n),
        'v4': np.random.normal(1.5, 1.5, n),
        'v5': np.random.normal(-1, 1.5, n),
        'is_fraud': 1
    })

df = pd.concat([gen_legit(N_LEGIT), gen_fraud(N_FRAUD)], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv('/home/claude/fraud_detection/data/transactions.csv', index=False)
print(f"Dataset: {len(df):,} rows | {df.is_fraud.sum():,} fraud ({df.is_fraud.mean()*100:.2f}%)")
print(f"Columns: {list(df.columns)}")
