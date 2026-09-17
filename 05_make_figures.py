import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("sweep_results.json") as f:
    data = json.load(f)

resA = data["sweepA"]
resB = data["sweepB"]
resC = data["sweepC"]
resE = data["sweepE"]
resF = data["sweepF"]

C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
C_GRID_E = [0.01, 0.1, 1.0, 10.0]
C_GRID_F = [0.01, 0.1, 1.0]


def agg(rows, key):
    vals = [r[key] for r in rows]
    return np.mean(vals), np.std(vals, ddof=1)


# --- Figure 1: margin, train/test accuracy, gap vs C (primal, both regimes), with error bars ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for overlap, color in [("low", "tab:blue"), ("high", "tab:orange")]:
    margin_m, margin_s = zip(*[agg([r for r in resA if r["overlap"] == overlap and r["C"] == C], "margin") for C in C_GRID])
    tr_m, tr_s = zip(*[agg([r for r in resA if r["overlap"] == overlap and r["C"] == C], "train_acc") for C in C_GRID])
    te_m, te_s = zip(*[agg([r for r in resA if r["overlap"] == overlap and r["C"] == C], "test_acc") for C in C_GRID])
    gap_m, gap_s = zip(*[agg([r for r in resA if r["overlap"] == overlap and r["C"] == C], "gap") for C in C_GRID])

    axes[0].errorbar(C_GRID, margin_m, yerr=margin_s, marker="o", color=color, label=f"{overlap} overlap", capsize=3)
    axes[1].errorbar(C_GRID, tr_m, yerr=tr_s, marker="o", color=color, linestyle="-", label=f"{overlap} train", capsize=3)
    axes[1].errorbar(C_GRID, te_m, yerr=te_s, marker="s", color=color, linestyle="--", label=f"{overlap} test", capsize=3)
    axes[2].errorbar(C_GRID, gap_m, yerr=gap_s, marker="o", color=color, label=f"{overlap} overlap", capsize=3)

for ax in axes:
    ax.set_xscale("log")
    ax.set_xlabel("C")
    ax.legend(fontsize=8)
axes[0].set_ylabel("Margin width  2/||w||")
axes[0].set_title("Margin vs C (H1)  [n=5 seeds, ±1 SD]")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Train/test accuracy vs C (H2)  [n=5 seeds, ±1 SD]")
axes[2].set_ylabel("Train - test accuracy")
axes[2].set_title("Generalisation gap vs C (H3)  [n=5 seeds, ±1 SD]")
plt.tight_layout()
plt.savefig("fig1_primal_C_sweep.png", dpi=130)
plt.close()

# --- Figure 2: SV ratio and bound-SV count vs C (dual, both regimes), with error bars ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for overlap, color in [("low", "tab:blue"), ("high", "tab:orange")]:
    sv_m, sv_s = zip(*[agg([r for r in resB if r["overlap"] == overlap and r["C"] == C], "sv_ratio") for C in C_GRID])
    nb_m, nb_s = zip(*[agg([r for r in resB if r["overlap"] == overlap and r["C"] == C], "n_bound") for C in C_GRID])
    axes[0].errorbar(C_GRID, sv_m, yerr=sv_s, marker="o", color=color, label=f"{overlap} overlap", capsize=3)
    axes[1].errorbar(C_GRID, nb_m, yerr=nb_s, marker="o", color=color, label=f"{overlap} overlap", capsize=3)
for ax in axes:
    ax.set_xscale("log")
    ax.set_xlabel("C")
    ax.legend(fontsize=8)
axes[0].set_ylabel("Support-vector ratio  n_SV/n_train")
axes[0].set_title("SV ratio vs C (H4)  [n=5 seeds, ±1 SD]")
axes[1].set_ylabel("Mean # bound SVs (alpha=C)")
axes[1].set_title("Bound SV count vs C (H4)  [n=5 seeds, ±1 SD]")
plt.tight_layout()
plt.savefig("fig2_dual_sv_counts.png", dpi=130)
plt.close()

# --- Figure 3: kernel comparison on moons, with error bars ---
fig, ax = plt.subplots(figsize=(5, 4))
for kernel, color in [("linear", "tab:red"), ("rbf", "tab:green")]:
    acc_m, acc_s = zip(*[agg([r for r in resC if r["kernel"] == kernel and r["C"] == C], "test_acc") for C in [0.1, 1.0, 10.0]])
    ax.errorbar([0.1, 1.0, 10.0], acc_m, yerr=acc_s, marker="o", color=color, label=kernel, capsize=3)
ax.set_xscale("log")
ax.set_xlabel("C")
ax.set_ylabel("Test accuracy")
ax.set_title("Linear vs RBF on moons (H5)  [n=5 seeds, ±1 SD]")
ax.legend()
plt.tight_layout()
plt.savefig("fig3_kernel_comparison_moons.png", dpi=130)
plt.close()

# --- Figure 4: real-dataset (Breast Cancer) C-sweep, with error bars ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
margin_m, margin_s = zip(*[agg([r for r in resE if r["C"] == C], "margin") for C in C_GRID_E])
tr_m, tr_s = zip(*[agg([r for r in resE if r["C"] == C], "train_acc") for C in C_GRID_E])
te_m, te_s = zip(*[agg([r for r in resE if r["C"] == C], "test_acc") for C in C_GRID_E])
gap_m, gap_s = zip(*[agg([r for r in resE if r["C"] == C], "gap") for C in C_GRID_E])
axes[0].errorbar(C_GRID_E, margin_m, yerr=margin_s, marker="o", color="tab:purple", capsize=3)
axes[1].errorbar(C_GRID_E, tr_m, yerr=tr_s, marker="o", color="tab:purple", linestyle="-", label="train", capsize=3)
axes[1].errorbar(C_GRID_E, te_m, yerr=te_s, marker="s", color="tab:purple", linestyle="--", label="test", capsize=3)
for ax in axes:
    ax.set_xscale("log")
    ax.set_xlabel("C")
axes[0].set_ylabel("Margin width  2/||w||")
axes[0].set_title("Breast Cancer: margin vs C  [n=5 splits, ±1 SD]")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Breast Cancer: train/test accuracy vs C  [n=5 splits, ±1 SD]")
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig("fig4_real_dataset_C_sweep.png", dpi=130)
plt.close()

print("Saved fig1_primal_C_sweep.png, fig2_dual_sv_counts.png, fig3_kernel_comparison_moons.png, fig4_real_dataset_C_sweep.png")

# --- Figure 5 (H6): support-vector composition vs C -- SV ratio, BVR, NDP ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for overlap, color in [("low", "tab:blue"), ("high", "tab:orange")]:
    sv_m, sv_s = zip(*[agg([r for r in resB if r["overlap"] == overlap and r["C"] == C], "sv_ratio") for C in C_GRID])
    bvr_m, bvr_s = zip(*[agg([r for r in resB if r["overlap"] == overlap and r["C"] == C], "bvr") for C in C_GRID])
    ndp_m, ndp_s = zip(*[agg([r for r in resB if r["overlap"] == overlap and r["C"] == C], "ndp") for C in C_GRID])
    axes[0].errorbar(C_GRID, sv_m, yerr=sv_s, marker="o", color=color, label=f"{overlap} overlap", capsize=3)
    axes[1].errorbar(C_GRID, bvr_m, yerr=bvr_s, marker="o", color=color, label=f"{overlap} overlap", capsize=3)
    axes[2].errorbar(C_GRID, ndp_m, yerr=ndp_s, marker="o", color=color, label=f"{overlap} overlap", capsize=3)

# Overlay the real-data composition over its narrower C range.
sv_m_f, sv_s_f = zip(*[agg([r for r in resF if r["C"] == C], "sv_ratio") for C in C_GRID_F])
bvr_m_f, bvr_s_f = zip(*[agg([r for r in resF if r["C"] == C], "bvr") for C in C_GRID_F])
ndp_m_f, ndp_s_f = zip(*[agg([r for r in resF if r["C"] == C], "ndp") for C in C_GRID_F])
axes[0].errorbar(C_GRID_F, sv_m_f, yerr=sv_s_f, marker="^", color="tab:green", linestyle="--", label="Breast Cancer (real)", capsize=3)
axes[1].errorbar(C_GRID_F, bvr_m_f, yerr=bvr_s_f, marker="^", color="tab:green", linestyle="--", label="Breast Cancer (real)", capsize=3)
axes[2].errorbar(C_GRID_F, ndp_m_f, yerr=ndp_s_f, marker="^", color="tab:green", linestyle="--", label="Breast Cancer (real)", capsize=3)

for ax in axes:
    ax.set_xscale("log")
    ax.set_xlabel("C")
    ax.legend(fontsize=8)
axes[0].set_ylabel("Support-vector ratio  n_SV/n_train")
axes[0].set_title("Total SV ratio vs C  [n=5 seeds/splits, ±1 SD]")
axes[1].set_ylabel("Bound-SV ratio (BVR)  n_bound/n_SV")
axes[1].set_title("Support-vector saturation (H6) vs C")
axes[2].set_ylabel("Normalised dual pressure (NDP)  mean(αᵢ/C)")
axes[2].set_title("Dual budget usage (H6) vs C")
plt.tight_layout()
plt.savefig("fig5_sv_composition.png", dpi=130)
plt.close()

print("Saved fig5_sv_composition.png")
