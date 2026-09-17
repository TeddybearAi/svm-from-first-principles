"""
Primal soft-margin SVM implemented from scratch with full-batch subgradient
descent.

Objective:
    J(w, b) = (1/2)||w||^2 + C * sum_i max(0, 1 - y_i (w.x_i + b))

Subgradient:
    dJ/dw = w - C * sum_{i: y_i f(x_i) < 1} y_i x_i
    dJ/db = -C * sum_{i: y_i f(x_i) < 1} y_i

The implementation supports fixed, decaying, and C-scaled learning rates and
records the objective value after each epoch.
"""

import numpy as np


class PrimalSVM:
    def __init__(self, C=1.0, lr=0.001, n_epochs=2000, lr_schedule="fixed", tol=1e-6, seed=0):
        """
        C: regularisation strength (higher C -> penalise margin violations more).
        lr: base learning rate.
        lr_schedule: "fixed", "decay" (lr / (1 + epoch * decay_rate)), or
            "c_scaled" (lr / max(1, C)) -- see note below.
        tol: stop early if relative change in objective falls below this for
             `patience` consecutive epochs (see fit()).

        For lr_schedule="c_scaled", the effective step size is divided by
        max(1, C) because the hinge-loss subgradient scales with C. This keeps
        update magnitudes more comparable across the C sweep.
        """
        self.C = C
        self.lr = lr
        self.n_epochs = n_epochs
        self.lr_schedule = lr_schedule
        self.tol = tol
        self.seed = seed
        self.w = None
        self.b = 0.0
        self.loss_history_ = []

    def _hinge_loss(self, X, y):
        margins = y * (X @ self.w + self.b)
        hinge = np.maximum(0, 1 - margins)
        return hinge

    def objective(self, X, y):
        return 0.5 * np.dot(self.w, self.w) + self.C * np.sum(self._hinge_loss(X, y))

    def decision_function(self, X):
        return X @ self.w + self.b

    def predict(self, X):
        # Explicit tie-break: an exact-zero score is assigned to the +1 class.
        return np.where(self.decision_function(X) >= 0, 1, -1)

    def fit(self, X, y):
        n, d = X.shape
        # Zero initialisation is sufficient for this convex linear objective.
        self.w = np.zeros(d)
        self.b = 0.0

        patience, stale = 20, 0
        prev_obj = None

        for epoch in range(self.n_epochs):
            lr_t = self.lr
            if self.lr_schedule == "decay":
                lr_t = self.lr / (1 + epoch * 1e-3)
            elif self.lr_schedule == "c_scaled":
                lr_t = self.lr / max(1.0, self.C)

            margins = y * (X @ self.w + self.b)
            violated = margins < 1  # boolean mask: inside margin or misclassified

            grad_w = self.w - self.C * (y[violated] @ X[violated]) if violated.any() else self.w.copy()
            grad_b = -self.C * y[violated].sum() if violated.any() else 0.0

            self.w -= lr_t * grad_w
            self.b -= lr_t * grad_b

            obj = self.objective(X, y)
            self.loss_history_.append(obj)

            if prev_obj is not None:
                rel_change = abs(prev_obj - obj) / max(abs(prev_obj), 1e-12)
                stale = stale + 1 if rel_change < self.tol else 0
                if stale >= patience:
                    break
            prev_obj = obj

        return self


def _load_data_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("data01", "01_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def verify_against_sklearn(model, ref, X, label=""):
    """Compare learned weights, bias, and decision scores with sklearn SVC."""
    scores_own = model.decision_function(X)
    scores_ref = ref.decision_function(X)
    w_own, w_ref = model.w, ref.coef_.ravel()
    cos_sim = np.dot(w_own, w_ref) / (np.linalg.norm(w_own) * np.linalg.norm(w_ref) + 1e-12)
    print(f"  [{label}] w cosine similarity={cos_sim:.5f}  "
          f"b diff={abs(model.b - ref.intercept_[0]):.5f}  "
          f"mean|Δscore|={np.mean(np.abs(scores_own - scores_ref)):.5f}  "
          f"max|Δscore|={np.max(np.abs(scores_own - scores_ref)):.5f}")


if __name__ == "__main__":
    from sklearn.svm import SVC

    data01 = _load_data_module()

    X, y = data01.make_synthetic(overlap="low")
    (Xtr, ytr), (Xval, yval), (Xte, yte), _ = data01.split_and_scale(X, y)

    model = PrimalSVM(C=1.0, lr=0.01, n_epochs=3000)
    model.fit(Xtr, ytr)
    train_acc = (model.predict(Xtr) == ytr).mean()
    test_acc = (model.predict(Xte) == yte).mean()
    print(f"[Synthetic low-overlap] train acc={train_acc:.3f} test acc={test_acc:.3f} "
          f"final objective={model.loss_history_[-1]:.4f} epochs run={len(model.loss_history_)}")

    ref = SVC(kernel="linear", C=1.0).fit(Xtr, ytr)
    ref_test_acc = ref.score(Xte, yte)
    print(f"  sklearn SVC (linear) reference test acc={ref_test_acc:.3f}")
    verify_against_sklearn(model, ref, Xte, label="Synthetic low-overlap")

    X2, y2 = data01.make_synthetic(overlap="high")
    (Xtr2, ytr2), (Xval2, yval2), (Xte2, yte2), _ = data01.split_and_scale(X2, y2)
    model2 = PrimalSVM(C=1.0, lr=0.01, n_epochs=3000)
    model2.fit(Xtr2, ytr2)
    print(f"[Synthetic high-overlap] train acc={(model2.predict(Xtr2)==ytr2).mean():.3f} "
          f"test acc={(model2.predict(Xte2)==yte2).mean():.3f}")
    ref2 = SVC(kernel="linear", C=1.0).fit(Xtr2, ytr2)
    verify_against_sklearn(model2, ref2, Xte2, label="Synthetic high-overlap")

    Xr, yr, _ = data01.load_real_dataset()
    (Xtr3, ytr3), (Xval3, yval3), (Xte3, yte3), _ = data01.split_and_scale(Xr, yr)
    model3 = PrimalSVM(C=1.0, lr=0.001, n_epochs=3000)
    model3.fit(Xtr3, ytr3)
    print(f"[Breast Cancer] train acc={(model3.predict(Xtr3)==ytr3).mean():.3f} "
          f"test acc={(model3.predict(Xte3)==yte3).mean():.3f}")
    ref3 = SVC(kernel="linear", C=1.0).fit(Xtr3, ytr3)
    print(f"  sklearn SVC (linear) reference test acc={ref3.score(Xte3, yte3):.3f}")
    verify_against_sklearn(model3, ref3, Xte3, label="Breast Cancer")
