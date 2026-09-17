"""
Data generation and preprocessing for the project.

Four experimental regimes are used:
  1) Low-overlap 2D Gaussian blobs.
  2) High-overlap 2D Gaussian blobs.
  3) Two interleaving half-moons for the nonlinear kernel comparison.
  4) Breast Cancer Wisconsin (569 samples, 30 features) for the real-data
     experiments.

All labels are mapped to {-1, +1}. Standardisation is fitted on the training
split only and then applied unchanged to validation and test data.
"""

import numpy as np
from sklearn.datasets import make_blobs, make_moons, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42


def make_synthetic(n_samples=300, overlap="low", seed=RANDOM_SEED):
    """
    Generate 2D binary-labelled Gaussian blobs with a controllable overlap level.

    overlap: "low" or "high" -> controls cluster_std relative to inter-centre distance.
    Labels returned as {-1, +1} (SVM convention), not {0, 1}.
    """
    centers = np.array([[-2.0, 0.0], [2.0, 0.0]])
    cluster_std = 1.0 if overlap == "low" else 2.6
    X, y = make_blobs(
        n_samples=n_samples,
        centers=centers,
        cluster_std=cluster_std,
        random_state=seed,
    )
    y = np.where(y == 0, -1, 1)
    return X, y


def make_nonlinear(n_samples=300, noise=0.15, seed=RANDOM_SEED):
    """Generate the nonlinear half-moons dataset used for the RBF comparison."""
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    y = np.where(y == 0, -1, 1)
    return X, y


def load_real_dataset():
    """
    Load Breast Cancer Wisconsin. Returns X (n, 30), y in {-1, +1}.
    sklearn encodes 0=malignant, 1=benign; we map malignant -> +1 (the class of
    primary clinical interest / "positive" class) and benign -> -1.
    """
    data = load_breast_cancer()
    X = data.data
    y = np.where(data.target == 0, 1, -1)  # malignant -> +1
    return X, y, list(data.feature_names)


def split_and_scale(X, y, test_size=0.2, val_size=0.2, seed=RANDOM_SEED, stratify=True):
    """
    Split into train/validation/test sets and standardise using training data only.

    val_size is a fraction of the original dataset, so test_size=0.2 and
    val_size=0.2 produce a 60/20/20 split.
    """
    strat = y if stratify else None
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=strat
    )
    val_fraction_of_temp = val_size / (1 - test_size)
    strat_temp = y_temp if stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_fraction_of_temp, random_state=seed, stratify=strat_temp
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    return (X_train_s, y_train), (X_val_s, y_val), (X_test_s, y_test), scaler


if __name__ == "__main__":
    X_syn_low, y_syn_low = make_synthetic(overlap="low")
    X_syn_high, y_syn_high = make_synthetic(overlap="high")
    print("Synthetic low-overlap:", X_syn_low.shape, np.bincount((y_syn_low + 1) // 2))
    print("Synthetic high-overlap:", X_syn_high.shape, np.bincount((y_syn_high + 1) // 2))

    X_real, y_real, feat_names = load_real_dataset()
    print("Real dataset:", X_real.shape, "classes:", np.bincount((y_real + 1) // 2), "n_features:", len(feat_names))

    (Xtr, ytr), (Xval, yval), (Xte, yte), scaler = split_and_scale(X_real, y_real)
    print("Split sizes -> train:", Xtr.shape, "val:", Xval.shape, "test:", Xte.shape)
    print("Train mean ~0, std ~1 check:", np.allclose(Xtr.mean(axis=0), 0, atol=1e-8), np.allclose(Xtr.std(axis=0), 1, atol=1e-6))
    for name, yy in [("train", ytr), ("val", yval), ("test", yte)]:
        pos_frac = (yy == 1).mean()
        print(f"{name} positive-class fraction: {pos_frac:.3f}")
