"""
Dual soft-margin SVM implemented from scratch with a simplified Sequential
Minimal Optimisation (SMO) routine.

Dual objective:
    max_alpha  sum_i alpha_i
               - 1/2 sum_i sum_j alpha_i alpha_j y_i y_j K(x_i, x_j)
    subject to 0 <= alpha_i <= C
               sum_i alpha_i y_i = 0

KKT interpretation used by the diagnostics:
    alpha_i = 0        -> y_i f(x_i) >= 1
    0 < alpha_i < C    -> y_i f(x_i) = 1
    alpha_i = C        -> y_i f(x_i) <= 1

Decision function:
    f(x) = sum_i alpha_i y_i K(x_i, x) + b
"""

import numpy as np


def linear_kernel(X, Z):
    return X @ Z.T


def rbf_kernel(X, Z, gamma=0.5):
    # ||x - z||^2 = ||x||^2 + ||z||^2 - 2 x.z, computed via broadcasting for a full
    # pairwise matrix without an explicit Python double loop.
    X_sq = np.sum(X ** 2, axis=1).reshape(-1, 1)
    Z_sq = np.sum(Z ** 2, axis=1).reshape(1, -1)
    sq_dists = X_sq + Z_sq - 2 * X @ Z.T
    sq_dists = np.maximum(sq_dists, 0)  # guard tiny negative values from float error
    return np.exp(-gamma * sq_dists)


class DualSVM_SMO:
    def __init__(self, C=1.0, kernel="linear", gamma=0.5, tol=1e-3, max_passes=10, sv_tol=1e-5, seed=0):
        self.C = C
        self.kernel_name = kernel
        self.gamma = gamma
        self.tol = tol
        self.max_passes = max_passes
        self.sv_tol = sv_tol  # numerical tolerance for deciding alpha_i > 0 (support vector)
        self.seed = seed

    def _kernel_fn(self, X, Z):
        if self.kernel_name == "linear":
            return linear_kernel(X, Z)
        elif self.kernel_name == "rbf":
            return rbf_kernel(X, Z, gamma=self.gamma)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel_name}")

    def fit(self, X, y):
        n = X.shape[0]
        rng = np.random.default_rng(self.seed)
        alphas = np.zeros(n)
        b = 0.0
        K = self._kernel_fn(X, X)  # precompute full Gram matrix (fine for small n)

        def f(i):
            return np.sum(alphas * y * K[:, i]) + b

        passes = 0
        while passes < self.max_passes:
            num_changed = 0
            for i in range(n):
                E_i = f(i) - y[i]
                if (y[i] * E_i < -self.tol and alphas[i] < self.C) or \
                   (y[i] * E_i > self.tol and alphas[i] > 0):
                    j = i
                    while j == i:
                        j = rng.integers(0, n)
                    E_j = f(j) - y[j]

                    alpha_i_old, alpha_j_old = alphas[i], alphas[j]
                    if y[i] != y[j]:
                        L = max(0, alphas[j] - alphas[i])
                        H = min(self.C, self.C + alphas[j] - alphas[i])
                    else:
                        L = max(0, alphas[i] + alphas[j] - self.C)
                        H = min(self.C, alphas[i] + alphas[j])
                    if L == H:
                        continue

                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue  # not a valid descent direction under this simplified scheme

                    alphas[j] = alphas[j] - (y[j] * (E_i - E_j)) / eta
                    alphas[j] = np.clip(alphas[j], L, H)

                    if abs(alphas[j] - alpha_j_old) < 1e-5:
                        continue

                    alphas[i] = alphas[i] + y[i] * y[j] * (alpha_j_old - alphas[j])

                    b1 = b - E_i - y[i] * (alphas[i] - alpha_i_old) * K[i, i] \
                         - y[j] * (alphas[j] - alpha_j_old) * K[i, j]
                    b2 = b - E_j - y[i] * (alphas[i] - alpha_i_old) * K[i, j] \
                         - y[j] * (alphas[j] - alpha_j_old) * K[j, j]

                    if 0 < alphas[i] < self.C:
                        b = b1
                    elif 0 < alphas[j] < self.C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2.0

                    num_changed += 1

            passes = passes + 1 if num_changed == 0 else 0

        self.alphas_ = alphas
        self.b_ = b
        self.X_fit_ = X
        self.y_fit_ = y
        sv_mask = alphas > self.sv_tol
        self.support_ = np.where(sv_mask)[0]
        self.n_support_ = sv_mask.sum()
        return self

    def decision_function(self, X):
        K = self._kernel_fn(self.X_fit_, X)  # (n_train, n_query)
        return (self.alphas_ * self.y_fit_) @ K + self.b_

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)

    def dual_objective(self):
        """Return the fitted model's dual objective value."""
        K = self._kernel_fn(self.X_fit_, self.X_fit_)
        quad = np.einsum("i,j,i,j,ij->", self.alphas_, self.alphas_, self.y_fit_, self.y_fit_, K)
        return np.sum(self.alphas_) - 0.5 * quad

    def kkt_report(self):
        """
        Return box/equality-constraint residuals and KKT margin diagnostics for
        non-support, free-support, and bound-support-vector regimes.
        """
        box_violation = np.maximum(0, -self.alphas_).max() + np.maximum(0, self.alphas_ - self.C).max()
        equality_residual = abs(np.dot(self.alphas_, self.y_fit_))
        margins = self.y_fit_ * self.decision_function(self.X_fit_)

        non_sv = self.alphas_ <= 1e-5
        free = (self.alphas_ > 1e-5) & (self.alphas_ < self.C - 1e-5)
        bound = self.alphas_ >= self.C - 1e-5

        # Worst-case margin violation for each alpha regime.
        non_sv_violation = np.maximum(0, 1 - margins[non_sv]).max() if non_sv.any() else 0.0
        free_residual = np.abs(margins[free] - 1).max() if free.any() else 0.0
        bound_violation = np.maximum(0, margins[bound] - 1).max() if bound.any() else 0.0

        return dict(
            box_violation=box_violation,
            equality_residual=equality_residual,
            free_sv_margin_residual=free_residual,
            non_sv_violation=non_sv_violation,
            bound_sv_violation=bound_violation,
            n_free=int(free.sum()),
            n_non_sv=int(non_sv.sum()),
            n_bound=int(bound.sum()),
        )


def _load_module(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


if __name__ == "__main__":
    from sklearn.svm import SVC
    data01 = _load_module("data01", "01_data.py")

    X, y = data01.make_synthetic(overlap="low")
    (Xtr, ytr), (Xval, yval), (Xte, yte), _ = data01.split_and_scale(X, y)

    dual_lin = DualSVM_SMO(C=1.0, kernel="linear", tol=1e-3, max_passes=20).fit(Xtr, ytr)
    acc = (dual_lin.predict(Xte) == yte).mean()
    ref = SVC(kernel="linear", C=1.0).fit(Xtr, ytr)
    print(f"[Dual-SMO linear, low overlap] test acc={acc:.3f}  n_SV={dual_lin.n_support_}  "
          f"| sklearn SVC test acc={ref.score(Xte, yte):.3f}  n_SV={len(ref.support_)}")
    kkt = dual_lin.kkt_report()
    scores_own, scores_ref = dual_lin.decision_function(Xte), ref.decision_function(Xte)
    print(f"  KKT/constraint check: box_violation={kkt['box_violation']:.2e}  "
          f"equality_residual|sum(alpha*y)|={kkt['equality_residual']:.2e}")
    print(f"  Per-regime margin residuals: non-SV (alpha=0) violation={kkt['non_sv_violation']:.2e} (n={kkt['n_non_sv']})  "
          f"free-SV (0<alpha<C) residual={kkt['free_sv_margin_residual']:.2e} (n={kkt['n_free']})  "
          f"bound-SV (alpha=C) violation={kkt['bound_sv_violation']:.2e} (n={kkt['n_bound']})")
    print(f"  decision-score agreement vs sklearn: mean|Δ|={np.mean(np.abs(scores_own-scores_ref)):.4f}  "
          f"max|Δ|={np.max(np.abs(scores_own-scores_ref)):.4f}  "
          f"own dual objective (monitored, not a standalone proof)={dual_lin.dual_objective():.4f}")

    X2, y2 = data01.make_synthetic(overlap="high")
    (Xtr2, ytr2), (Xval2, yval2), (Xte2, yte2), _ = data01.split_and_scale(X2, y2)

    gamma = 0.5
    dual_rbf = DualSVM_SMO(C=1.0, kernel="rbf", gamma=gamma, tol=1e-3, max_passes=20).fit(Xtr2, ytr2)
    acc_rbf = (dual_rbf.predict(Xte2) == yte2).mean()
    ref_rbf = SVC(kernel="rbf", C=1.0, gamma=gamma).fit(Xtr2, ytr2)
    print(f"[Dual-SMO RBF, high overlap] test acc={acc_rbf:.3f}  n_SV={dual_rbf.n_support_}  "
          f"| sklearn SVC(rbf) test acc={ref_rbf.score(Xte2, yte2):.3f}  n_SV={len(ref_rbf.support_)}")

    Xr, yr, _ = data01.load_real_dataset()
    (Xtr3, ytr3), (Xval3, yval3), (Xte3, yte3), _ = data01.split_and_scale(Xr, yr)
    dual_real = DualSVM_SMO(C=1.0, kernel="linear", tol=1e-3, max_passes=15).fit(Xtr3, ytr3)
    acc3 = (dual_real.predict(Xte3) == yte3).mean()
    ref3 = SVC(kernel="linear", C=1.0).fit(Xtr3, ytr3)
    print(f"[Dual-SMO linear, Breast Cancer] test acc={acc3:.3f}  n_SV={dual_real.n_support_}  "
          f"| sklearn SVC test acc={ref3.score(Xte3, yte3):.3f}  n_SV={len(ref3.support_)}")
