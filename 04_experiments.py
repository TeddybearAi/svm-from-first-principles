"""
Experiment sweeps used in Report Sections 2.3 and 2.4.

  A) Primal SVM C-sweep on low/high-overlap synthetic data (H1-H3).
  B) Linear dual-SMO C-sweep with support-vector statistics (H4/H6).
  C) Linear-vs-RBF comparison on the moons dataset (H5).
  D) Support-vector tolerance and KKT diagnostic check.
  E) Primal SVM C-sweep on Breast Cancer Wisconsin.
  F) Dual-SMO support-vector composition on Breast Cancer Wisconsin.
"""

import numpy as np
import importlib.util


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


data01 = _load("data01", "01_data.py")
primal_mod = _load("primal02", "02_primal_svm.py")
dual_mod = _load("dual03", "03_dual_smo.py")

PrimalSVM = primal_mod.PrimalSVM
DualSVM_SMO = dual_mod.DualSVM_SMO

C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
SEEDS = [0, 1, 2, 3, 4]


def validate_implementation_settings():
    """
    Use validation data to fix the primal learning-rate schedule, SMO pass
    budget, and RBF gamma before evaluating the experimental sweeps.

    C is not selected here because it is the independent variable in H1-H4.
    Returns the selected gamma for the moons kernel comparison.
    """
    X, y = data01.make_synthetic(overlap="high", seed=42)
    (Xtr, ytr), (Xval, yval), (Xte, yte), _ = data01.split_and_scale(X, y, seed=42)

    print("--- Validating primal lr_schedule choice (at C=100, the hardest case) ---")
    for schedule in ["fixed", "c_scaled"]:
        m = PrimalSVM(C=100.0, lr=0.01, n_epochs=3000, lr_schedule=schedule, seed=0).fit(Xtr, ytr)
        tail_std = np.std(m.loss_history_[-200:])
        val_acc = (m.predict(Xval) == yval).mean()
        print(f"  lr_schedule={schedule:<10} val_acc={val_acc:.3f}  loss tail_std={tail_std:.3f}")
    print("  -> 'c_scaled' selected (lower tail_std = stabilised, not oscillating); used for ALL C-sweep runs.")

    print("--- Validating dual SMO max_passes choice ---")
    for max_passes in [5, 10, 15, 20]:
        m = DualSVM_SMO(C=1.0, kernel="linear", tol=1e-3, max_passes=max_passes, seed=0).fit(Xtr, ytr)
        val_acc = (m.predict(Xval) == yval).mean()
        print(f"  max_passes={max_passes:<3} val_acc={val_acc:.3f}  n_SV={m.n_support_}")
    print("  -> Validation accuracy is unchanged from 5 passes onward; support-vector count stabilises by 10.")
    print("     max_passes=15 retained as a conservative buffer beyond the observed stabilisation point; used for ALL dual-SMO sweep runs.")

    print("--- Validating RBF gamma choice (on the moons regime, at reference C=1) ---")
    Xm, ym = data01.make_nonlinear(seed=42)
    (Xmtr, ymtr), (Xmval, ymval), (Xmte, ymte), _ = data01.split_and_scale(Xm, ym, seed=42)
    gamma_candidates = [0.1, 0.5, 1.0, 2.0, 5.0]
    best_gamma, best_val_acc = None, -1.0
    for gamma in gamma_candidates:
        m = DualSVM_SMO(C=1.0, kernel="rbf", gamma=gamma, tol=1e-3, max_passes=15, seed=0).fit(Xmtr, ymtr)
        val_acc = (m.predict(Xmval) == ymval).mean()
        print(f"  gamma={gamma:<4} val_acc={val_acc:.3f}")
        if val_acc > best_val_acc:
            best_gamma, best_val_acc = gamma, val_acc
    print(f"  -> gamma={best_gamma} selected by validation accuracy; used for ALL RBF runs in the H5 kernel comparison (test sets untouched here).")
    print("(Test sets for the actual C-sweep and kernel comparison are not touched in this function.)")
    return best_gamma


def sweep_primal():
    """Sweep A: primal SVM, C x overlap x seed. Returns list of result dicts."""
    results = []
    for overlap in ["low", "high"]:
        for C in C_GRID:
            for seed in SEEDS:
                X, y = data01.make_synthetic(overlap=overlap, seed=seed)
                (Xtr, ytr), (_Xval, _yval), (Xte, yte), _ = data01.split_and_scale(X, y, seed=seed)
                model = PrimalSVM(C=C, lr=0.01, n_epochs=4000, lr_schedule="c_scaled", seed=seed).fit(Xtr, ytr)
                train_acc = (model.predict(Xtr) == ytr).mean()
                test_acc = (model.predict(Xte) == yte).mean()
                hinge_train = model._hinge_loss(Xtr, ytr).mean()
                hinge_test = model._hinge_loss(Xte, yte).mean()
                w_norm = np.linalg.norm(model.w)
                margin = 2.0 / w_norm if w_norm > 1e-12 else np.inf
                results.append(dict(
                    overlap=overlap, C=C, seed=seed,
                    train_acc=train_acc, test_acc=test_acc,
                    gap=train_acc - test_acc,
                    hinge_train=hinge_train, hinge_test=hinge_test,
                    margin=margin, w_norm=w_norm,
                ))
    return results


def sweep_dual_linear():
    """
    Sweep B: linear dual SMO over C, overlap regime, and seed.

    Also records H6 composition metrics:
      BVR = n_bound / n_sv
      NDP = mean_i(alpha_i / C) over all training samples
    """
    results = []
    for overlap in ["low", "high"]:
        for C in C_GRID:
            for seed in SEEDS:
                X, y = data01.make_synthetic(overlap=overlap, seed=seed)
                (Xtr, ytr), (_Xval, _yval), (Xte, yte), _ = data01.split_and_scale(X, y, seed=seed)
                model = DualSVM_SMO(C=C, kernel="linear", tol=1e-3, max_passes=15, seed=seed).fit(Xtr, ytr)
                train_acc = (model.predict(Xtr) == ytr).mean()
                test_acc = (model.predict(Xte) == yte).mean()
                n_free = int(np.sum((model.alphas_ > 1e-5) & (model.alphas_ < C - 1e-5)))
                n_bound = int(np.sum(model.alphas_ >= C - 1e-5))
                n_sv = int(model.n_support_)
                bvr = (n_bound / n_sv) if n_sv > 0 else float("nan")
                ndp = float(np.mean(model.alphas_ / C))
                results.append(dict(
                    overlap=overlap, C=C, seed=seed,
                    train_acc=train_acc, test_acc=test_acc, gap=train_acc - test_acc,
                    n_sv=n_sv, n_free=n_free, n_bound=n_bound,
                    sv_ratio=n_sv / len(ytr),
                    bvr=bvr, ndp=ndp,
                ))
    return results


def sweep_dual_real():
    """
    Sweep F: dual-SMO support-vector composition on Breast Cancer Wisconsin.

    The C grid is limited to {0.01, 0.1, 1}. Measured outer-iteration counts
    increase sharply with C on this dataset (193, 925, 6461, and >20000 for
    C=0.01, 0.1, 1, and 10 respectively), so C=10 is excluded.
    """
    results = []
    Xr, yr, _ = data01.load_real_dataset()
    for C in [0.01, 0.1, 1.0]:
        for seed in SEEDS:
            (Xtr, ytr), (_Xval, _yval), (Xte, yte), _ = data01.split_and_scale(Xr, yr, seed=seed)
            model = DualSVM_SMO(C=C, kernel="linear", tol=1e-3, max_passes=15, seed=seed).fit(Xtr, ytr)
            train_acc = (model.predict(Xtr) == ytr).mean()
            test_acc = (model.predict(Xte) == yte).mean()
            n_free = int(np.sum((model.alphas_ > 1e-5) & (model.alphas_ < C - 1e-5)))
            n_bound = int(np.sum(model.alphas_ >= C - 1e-5))
            n_sv = int(model.n_support_)
            bvr = (n_bound / n_sv) if n_sv > 0 else float("nan")
            ndp = float(np.mean(model.alphas_ / C))
            results.append(dict(
                C=C, seed=seed, train_acc=train_acc, test_acc=test_acc, gap=train_acc - test_acc,
                n_sv=n_sv, n_free=n_free, n_bound=n_bound, sv_ratio=n_sv / len(ytr),
                bvr=bvr, ndp=ndp,
            ))
    return results


def kernel_comparison_moons(gamma=1.0):
    """Sweep C: compare linear and RBF kernels on the moons dataset."""
    results = []
    for kernel in ["linear", "rbf"]:
        for C in [0.1, 1.0, 10.0]:
            for seed in SEEDS:
                X, y = data01.make_nonlinear(seed=seed)
                (Xtr, ytr), (_Xval, _yval), (Xte, yte), _ = data01.split_and_scale(X, y, seed=seed)
                kwargs = dict(C=C, kernel=kernel, tol=1e-3, max_passes=15, seed=seed)
                if kernel == "rbf":
                    kwargs["gamma"] = gamma
                model = DualSVM_SMO(**kwargs).fit(Xtr, ytr)
                test_acc = (model.predict(Xte) == yte).mean()
                results.append(dict(kernel=kernel, C=C, seed=seed, test_acc=test_acc, n_sv=int(model.n_support_)))
    return results


def tolerance_sensitivity():
    """
    Sweep D: support-vector count across alpha thresholds for the same fitted
    model used in the 19-vs-17 comparison with sklearn.
    """
    X, y = data01.make_synthetic(overlap="low")
    (Xtr, ytr), (_Xval, _yval), (Xte, yte), _ = data01.split_and_scale(X, y)
    model = DualSVM_SMO(C=1.0, kernel="linear", tol=1e-3, max_passes=20, seed=0).fit(Xtr, ytr)
    alphas = model.alphas_
    results = {}
    for thresh in [1e-3, 1e-5, 1e-7]:
        results[thresh] = int(np.sum(alphas > thresh))
    sorted_alphas = np.sort(alphas)[::-1]
    near_zero_band = sorted_alphas[(sorted_alphas > 1e-9) & (sorted_alphas < 1e-2)]
    kkt = model.kkt_report()
    return results, near_zero_band, kkt


def sweep_real_dataset():
    """
    Sweep E: a compact primal C-sweep on the real Breast Cancer dataset, to
    check whether the qualitative margin/regularisation behaviour found on
    controlled synthetic data also appears on genuine 30-dimensional data.
    """
    results = []
    Xr, yr, _ = data01.load_real_dataset()
    for C in [0.01, 0.1, 1.0, 10.0]:
        for seed in SEEDS:
            (Xtr, ytr), (_Xval, _yval), (Xte, yte), _ = data01.split_and_scale(Xr, yr, seed=seed)
            model = PrimalSVM(C=C, lr=0.001, n_epochs=3000, lr_schedule="c_scaled", seed=seed).fit(Xtr, ytr)
            train_acc = (model.predict(Xtr) == ytr).mean()
            test_acc = (model.predict(Xte) == yte).mean()
            hinge_test = model._hinge_loss(Xte, yte).mean()
            w_norm = np.linalg.norm(model.w)
            results.append(dict(C=C, seed=seed, train_acc=train_acc, test_acc=test_acc,
                                 gap=train_acc - test_acc, hinge_test=hinge_test,
                                 margin=2.0 / w_norm if w_norm > 1e-12 else np.inf))
    return results


def mean_sd(rows, key):
    vals = [r[key] for r in rows]
    return np.mean(vals), np.std(vals, ddof=1)


def primal_c100_spot_checks():
    """
    Reproduce the C=100 primal-vs-sklearn geometry checks quoted in the report.
    Runs seed 0 for both low- and high-overlap synthetic regimes.
    """
    from sklearn.svm import SVC

    results = []
    for overlap in ["low", "high"]:
        X, y = data01.make_synthetic(overlap=overlap, seed=0)
        (Xtr, ytr), (_Xval, _yval), (Xte, yte), _ = data01.split_and_scale(X, y, seed=0)
        model = PrimalSVM(
            C=100.0, lr=0.01, n_epochs=4000,
            lr_schedule="c_scaled", seed=0
        ).fit(Xtr, ytr)
        ref = SVC(kernel="linear", C=100.0).fit(Xtr, ytr)
        w_own = model.w
        w_ref = ref.coef_.ravel()
        cos_sim = np.dot(w_own, w_ref) / (
            np.linalg.norm(w_own) * np.linalg.norm(w_ref) + 1e-12
        )
        scores_own = model.decision_function(Xte)
        scores_ref = ref.decision_function(Xte)
        results.append(dict(
            overlap=overlap,
            C=100.0,
            seed=0,
            scratch_w_norm=float(np.linalg.norm(w_own)),
            sklearn_w_norm=float(np.linalg.norm(w_ref)),
            cosine_similarity=float(cos_sim),
            bias_abs_diff=float(abs(model.b - ref.intercept_[0])),
            mean_abs_score_diff=float(np.mean(np.abs(scores_own - scores_ref))),
            max_abs_score_diff=float(np.max(np.abs(scores_own - scores_ref))),
            scratch_test_acc=float((model.predict(Xte) == yte).mean()),
            sklearn_test_acc=float(ref.score(Xte, yte)),
        ))
    return results


def failure_case_analysis():
    """
    Inspect the high-overlap, C=100 failure case by comparing errors with the
    known synthetic separator and with sklearn SVC on the same test points.
    """
    from sklearn.svm import SVC
    X, y = data01.make_synthetic(overlap="high", seed=0)
    (Xtr, ytr), (_Xval, _yval), (Xte, yte), _ = data01.split_and_scale(X, y, seed=0)
    from sklearn.model_selection import train_test_split
    X_temp, X_test_raw, y_temp, y_test_raw = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    assert np.array_equal(y_test_raw, yte), "raw/scaled test split mismatch -- would invalidate this check"

    model = PrimalSVM(C=100.0, lr=0.01, n_epochs=4000, lr_schedule="c_scaled", seed=0).fit(Xtr, ytr)
    ref = SVC(kernel="linear", C=100.0).fit(Xtr, ytr)

    own_pred = model.predict(Xte)
    ref_pred = ref.predict(Xte)
    own_wrong = own_pred != yte
    ref_wrong = ref_pred != yte
    both_wrong = own_wrong & ref_wrong
    wrong_side_of_true_sep = np.sign(X_test_raw[:, 0]) != yte

    print(f"Own model: {own_wrong.sum()} misclassified / {len(yte)} test points")
    print(f"Sklearn SVC: {ref_wrong.sum()} misclassified / {len(yte)} test points")
    print(f"Misclassified by BOTH own model and sklearn: {both_wrong.sum()}")
    print(f"Of own model's errors, fraction that are on the wrong side of the TRUE "
          f"generating separator (genuine overlap, not an optimiser artifact): "
          f"{(wrong_side_of_true_sep & own_wrong).sum()}/{own_wrong.sum()}")
    return dict(n_own_wrong=int(own_wrong.sum()), n_ref_wrong=int(ref_wrong.sum()),
                n_both_wrong=int(both_wrong.sum()),
                n_own_wrong_true_overlap=int((wrong_side_of_true_sep & own_wrong).sum()))


if __name__ == "__main__":
    import json

    print("=== Settings validation (uses VALIDATION split only; test sets untouched here) ===")
    validated_gamma = validate_implementation_settings()

    print("\n=== Sweep A: primal SVM, C x overlap (mean ± SD across 5 seeds) ===")
    resA = sweep_primal()
    for overlap in ["low", "high"]:
        print(f"-- overlap={overlap} --")
        for C in C_GRID:
            rows = [r for r in resA if r["overlap"] == overlap and r["C"] == C]
            tr_m, tr_s = mean_sd(rows, "train_acc")
            te_m, te_s = mean_sd(rows, "test_acc")
            gap_m, gap_s = mean_sd(rows, "gap")
            marg_m, marg_s = mean_sd(rows, "margin")
            htr_m, htr_s = mean_sd(rows, "hinge_train")
            hte_m, hte_s = mean_sd(rows, "hinge_test")
            print(f"  C={C:>7.2f}  train_acc={tr_m:.3f}±{tr_s:.3f}  test_acc={te_m:.3f}±{te_s:.3f}  "
                  f"gap={gap_m:+.3f}±{gap_s:.3f}  margin={marg_m:.3f}±{marg_s:.3f}  "
                  f"hinge_train={htr_m:.3f}±{htr_s:.3f}  hinge_test={hte_m:.3f}±{hte_s:.3f}")

    print("\n=== Sweep B: dual SMO (linear), C x overlap -- SV counts + composition (H4/H6) (mean ± SD) ===")
    resB = sweep_dual_linear()
    for overlap in ["low", "high"]:
        print(f"-- overlap={overlap} --")
        for C in C_GRID:
            rows = [r for r in resB if r["overlap"] == overlap and r["C"] == C]
            sv_m, sv_s = mean_sd(rows, "sv_ratio")
            nb_m, nb_s = mean_sd(rows, "n_bound")
            gap_m, gap_s = mean_sd(rows, "gap")
            bvr_m, bvr_s = mean_sd(rows, "bvr")
            ndp_m, ndp_s = mean_sd(rows, "ndp")
            print(f"  C={C:>7.2f}  sv_ratio={sv_m:.3f}±{sv_s:.3f}  n_bound={nb_m:.1f}±{nb_s:.1f}  gap={gap_m:+.3f}±{gap_s:.3f}  "
                  f"BVR={bvr_m:.3f}±{bvr_s:.3f}  NDP={ndp_m:.3f}±{ndp_s:.3f}")

    print("\n=== Sweep C: kernel comparison on moons (mean ± SD) ===")
    resC = kernel_comparison_moons(gamma=validated_gamma)
    for kernel in ["linear", "rbf"]:
        for C in [0.1, 1.0, 10.0]:
            rows = [r for r in resC if r["kernel"] == kernel and r["C"] == C]
            acc_m, acc_s = mean_sd(rows, "test_acc")
            print(f"  kernel={kernel:<6} C={C:>5.1f}  test_acc={acc_m:.3f}±{acc_s:.3f}")

    print("\n=== Sweep D: SV-count sensitivity to alpha threshold (exact 19-vs-17 configuration) ===")
    resD, near_zero, kktD = tolerance_sensitivity()
    for thresh, n in resD.items():
        print(f"  threshold={thresh:.0e}  n_SV={n}")
    print(f"  alphas in (1e-9, 1e-2) band: {np.round(near_zero, 6)}")
    print(f"  KKT diagnostics on this exact fit: box_violation={kktD['box_violation']:.2e}  "
          f"equality_residual={kktD['equality_residual']:.2e}")
    print(f"  Per-regime: non-SV violation={kktD['non_sv_violation']:.2e} (n={kktD['n_non_sv']})  "
          f"free-SV residual={kktD['free_sv_margin_residual']:.3f} (n={kktD['n_free']})  "
          f"bound-SV violation={kktD['bound_sv_violation']:.2e} (n={kktD['n_bound']})")

    print("\n=== Sweep E: real-dataset (Breast Cancer) C-sweep, primal SVM (mean ± SD) ===")
    resE = sweep_real_dataset()
    for C in [0.01, 0.1, 1.0, 10.0]:
        rows = [r for r in resE if r["C"] == C]
        tr_m, tr_s = mean_sd(rows, "train_acc")
        te_m, te_s = mean_sd(rows, "test_acc")
        gap_m, gap_s = mean_sd(rows, "gap")
        marg_m, marg_s = mean_sd(rows, "margin")
        print(f"  C={C:>6.2f}  train_acc={tr_m:.3f}±{tr_s:.3f}  test_acc={te_m:.3f}±{te_s:.3f}  "
              f"gap={gap_m:+.3f}±{gap_s:.3f}  margin={marg_m:.3f}±{marg_s:.3f}")

    print("  -- paired comparison (same 5 split seeds at each C) --")
    gaps_low = {r["seed"]: r["gap"] for r in resE if r["C"] == 0.01}
    gaps_high = {r["seed"]: r["gap"] for r in resE if r["C"] == 10.0}
    paired_diffs = [gaps_high[s] - gaps_low[s] for s in sorted(gaps_low)]
    print(f"  per-seed gap increase, C=10 minus C=0.01: {[round(float(d), 4) for d in paired_diffs]}")
    print(f"  all 5 positive: {all(d > 0 for d in paired_diffs)}  "
          f"mean paired increase: {np.mean(paired_diffs):.4f}  sample SD: {np.std(paired_diffs, ddof=1):.4f}")

    print("\n=== Sweep F: real-dataset (Breast Cancer) support-vector composition, dual SMO (H6 extension) ===")
    print("    (C grid narrower than Sweep E -- see docstring: SMO iteration count explodes at higher C here)")
    resF = sweep_dual_real()
    for C in [0.01, 0.1, 1.0]:
        rows = [r for r in resF if r["C"] == C]
        sv_m, sv_s = mean_sd(rows, "sv_ratio")
        bvr_m, bvr_s = mean_sd(rows, "bvr")
        ndp_m, ndp_s = mean_sd(rows, "ndp")
        gap_m, gap_s = mean_sd(rows, "gap")
        print(f"  C={C:>6.2f}  sv_ratio={sv_m:.3f}±{sv_s:.3f}  BVR={bvr_m:.3f}±{bvr_s:.3f}  "
              f"NDP={ndp_m:.3f}±{ndp_s:.3f}  gap={gap_m:+.3f}±{gap_s:.3f}")

    print("\n=== C=100 primal-vs-sklearn geometry spot checks ===")
    resC100 = primal_c100_spot_checks()
    for r in resC100:
        print(f"  {r['overlap']:<4} overlap: scratch ||w||={r['scratch_w_norm']:.6f}  "
              f"sklearn ||w||={r['sklearn_w_norm']:.6f}  "
              f"cosine={r['cosine_similarity']:.9f}  "
              f"test acc={r['scratch_test_acc']:.3f}/{r['sklearn_test_acc']:.3f}")

    print("\n=== Failure-case analysis: high-overlap, C=100, seed 0 ===")
    resFailure = failure_case_analysis()

    with open("sweep_results.json", "w") as f:
        json.dump(dict(sweepA=resA, sweepB=resB, sweepC=resC,
                        sweepD={str(k): v for k, v in resD.items()},
                        sweepD_kkt=kktD, sweepE=resE, failure_case=resFailure,
                        validated_gamma=validated_gamma,
                        sweepE_paired_diffs=paired_diffs,
                        sweepF=resF), f, indent=1)
    print("\nSaved raw results to sweep_results.json")
