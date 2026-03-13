import numpy as np

def smote_oversample(X, y, target_ratio=0.5, k=5, random_state=42):
    """Manual SMOTE implementation - no imblearn needed."""
    rng = np.random.RandomState(random_state)
    X_min = X[y == 1]
    X_maj = X[y == 0]
    n_min = len(X_min)
    n_maj = len(X_maj)
    n_needed = int(n_maj * target_ratio) - n_min
    if n_needed <= 0:
        return X, y

    synthetic = []
    for _ in range(n_needed):
        idx = rng.randint(0, n_min)
        sample = X_min[idx]
        # Find k nearest neighbors in minority class
        dists = np.sum((X_min - sample) ** 2, axis=1)
        dists[idx] = np.inf
        nn_idx = np.argpartition(dists, min(k, n_min-1))[:k]
        neighbor = X_min[rng.choice(nn_idx)]
        lam = rng.random()
        synthetic.append(sample + lam * (neighbor - sample))

    X_syn = np.array(synthetic)
    y_syn = np.ones(len(synthetic))
    X_out = np.vstack([X, X_syn])
    y_out = np.concatenate([y, y_syn])
    # Shuffle
    perm = rng.permutation(len(X_out))
    return X_out[perm], y_out[perm]
