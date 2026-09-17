# SVM from First Principles

From-scratch Support Vector Machine project for UTS Machine Learning A2/A3.

The project studies how the soft-margin parameter `C`, class overlap, kernel choice, and support-vector composition affect SVM behaviour. It includes a primal soft-margin SVM trained with full-batch subgradient descent and a simplified dual SMO solver with linear and RBF kernels.

## Repository structure

- `01_data.py` — data generation, Breast Cancer Wisconsin loading, train/validation/test splitting and standardisation.
- `02_primal_svm.py` — primal soft-margin SVM implemented from scratch.
- `03_dual_smo.py` — simplified dual SMO SVM implemented from scratch.
- `04_experiments.py` — validation checks and experiment sweeps A–F.
- `05_make_figures.py` — reproduces Figures 1–5 from the saved sweep results.
- `SVM_Project_Notebook.ipynb` — notebook version of the complete implementation and experiments.
- `sweep_results.json` — raw experimental outputs used for tables and figures.
- `c100_primal_verification.json` — recorded C=100 scratch-vs-sklearn geometry spot checks used in the report.

## Main experiments

1. Primal C-sweep on low- and high-overlap Gaussian data.
2. Dual SMO C-sweep with support-vector statistics.
3. Linear-vs-RBF comparison on the nonlinear moons dataset.
4. Numerical tolerance and KKT diagnostics.
5. Real-data primal C-sweep on Breast Cancer Wisconsin.
6. Real-data support-vector composition analysis using BVR and NDP.

`sklearn.svm.SVC` is used only as an external verification baseline; the reported primal and dual solvers are implemented from scratch.

## Running locally

Install the dependencies in `requirements.txt`, then run:

```bash
python 04_experiments.py
python 05_make_figures.py
```

The experiment script writes `sweep_results.json`; the plotting script reads that file and produces Figures 1–5.

The public Colab link will be added here once the final notebook is published.
