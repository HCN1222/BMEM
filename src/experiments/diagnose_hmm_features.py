#!/usr/bin/env python3
"""
Diagnose numerical issues in HMM training features and existing scan results.

Part A (feature statistics from a hmm_data_train.npz):
- per-dimension variance / scale comparison
- zero-value ratios and all-zero observation days
- feature correlation matrix (collinearity check)
- discrete value distribution of c_t

Part B (optional, from evaluate_states.py output directories via --exp-dir):
- per-state covariance health: min eigenvalue, condition number
- log-likelihood / BIC summary per state count
"""

import argparse
import json
from pathlib import Path

import numpy as np

from src.utils.paths import add_broker_path_args, paths_from_args


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose feature collinearity, degenerate variance, and covariance health for HMM training."
    )
    add_broker_path_args(parser)
    parser.add_argument("--input-file", "--input_file", type=Path, help="Override the broker HMM training .npz file.")
    parser.add_argument("--exp-dir", "--exp_dir", type=Path, action="append", default=[],
                        help="evaluate_states.py output directory (states_*/ subdirs). Repeatable.")
    return parser.parse_args()


def load_observations(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    observations = np.asarray(data["observations"], dtype=float)
    feature_names = np.asarray(data["feature_names"]).astype(str)
    return observations, feature_names


def print_matrix(matrix: np.ndarray, labels, title: str, fmt: str = "{:>12.4f}"):
    print(f"\n{title}")
    header = " " * 8 + "".join(f"{name:>12}" for name in labels)
    print(header)
    for i, name in enumerate(labels):
        row = "".join(fmt.format(v) for v in matrix[i])
        print(f"{name:>8}{row}")


def diagnose_features(observations: np.ndarray, feature_names):
    n_obs, n_features = observations.shape
    print("=" * 70)
    print(" PART A: FEATURE STATISTICS")
    print("=" * 70)
    print(f"observations: {n_obs} x {n_features}")

    print(f"\n{'feature':>8}{'mean':>14}{'variance':>14}{'std':>14}{'min':>10}{'max':>10}{'zero%':>8}")
    variances = observations.var(axis=0)
    for i, name in enumerate(feature_names):
        col = observations[:, i]
        zero_ratio = np.mean(col == 0.0) * 100
        print(f"{name:>8}{col.mean():>14.6f}{variances[i]:>14.6f}{col.std():>14.6f}"
              f"{col.min():>10.3f}{col.max():>10.3f}{zero_ratio:>8.2f}")

    scale_ratio = variances.max() / max(variances.min(), 1e-300)
    print(f"\nvariance scale ratio (max/min): {scale_ratio:.1f}")

    all_zero = np.mean(np.all(observations == 0.0, axis=1)) * 100
    print(f"all-zero observation rows: {all_zero:.2f}%")

    corr = np.corrcoef(observations, rowvar=False)
    print_matrix(corr, feature_names, "correlation matrix:")

    eigvals = np.linalg.eigvalsh(corr)
    print(f"\ncorrelation matrix eigenvalues: {np.array2string(eigvals, precision=4)}")
    print(f"condition number of correlation matrix: {eigvals.max() / eigvals.min():.1f}")

    cov = np.cov(observations, rowvar=False)
    cov_eigvals = np.linalg.eigvalsh(cov)
    print(f"\nglobal covariance eigenvalues: {np.array2string(cov_eigvals, precision=8)}")
    print(f"condition number of global covariance: {cov_eigvals.max() / max(cov_eigvals.min(), 1e-300):.1e}")

    if "c_t" in list(feature_names):
        idx = list(feature_names).index("c_t")
        values, counts = np.unique(observations[:, idx], return_counts=True)
        print("\nc_t value distribution:")
        for v, c in zip(values, counts):
            print(f"  c_t = {v:>5.1f}: {c:>8d} ({c / n_obs * 100:.2f}%)")


def bic_from_loglik(log_likelihood: float, n_states: int, n_features: int, covariance_type: str, n_samples: int):
    cov_params = n_states * n_features if covariance_type == "diag" else n_states * (n_features * (n_features + 1)) / 2
    n_params = n_states * (n_states - 1) + (n_states - 1) + n_states * n_features + cov_params
    return -2 * log_likelihood + n_params * np.log(n_samples)


def diagnose_exp_dir(exp_dir: Path, n_samples: int, feature_names):
    print("\n" + "=" * 70)
    print(f" PART B: COVARIANCE HEALTH — {exp_dir}")
    print("=" * 70)

    state_dirs = sorted(exp_dir.glob("states_*"), key=lambda p: int(p.name.split("_")[1]))
    if not state_dirs:
        print("no states_* subdirectories found")
        return

    print(f"\n{'states':>7}{'cov_type':>9}{'iters':>7}{'final loglik':>16}{'BIC':>16}"
          f"{'min eig':>12}{'max cond':>12}{'min diag var':>14}")

    for state_dir in state_dirs:
        meta_path = state_dir / "metadata.json"
        params_path = state_dir / "trained_hmm_params.npz"
        if not meta_path.exists() or not params_path.exists():
            continue
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        n_states = meta["n_states"]
        cov_type = meta["covariance_type"]
        loglik = meta["final_log_likelihood"]
        bic = bic_from_loglik(loglik, n_states, meta["n_features"], cov_type, n_samples)

        params = np.load(params_path, allow_pickle=True)
        covars = params["covars"]

        min_eig, max_cond, min_diag = np.inf, 0.0, np.inf
        for i in range(n_states):
            cov_i = covars[i] if covars[i].ndim == 2 else np.diag(covars[i])
            eigvals = np.linalg.eigvalsh((cov_i + cov_i.T) / 2)
            min_eig = min(min_eig, eigvals.min())
            max_cond = max(max_cond, eigvals.max() / max(eigvals.min(), 1e-300))
            min_diag = min(min_diag, np.diag(cov_i).min())

        print(f"{n_states:>7}{cov_type:>9}{meta['trained_iterations']:>7}{loglik:>16.2f}{bic:>16.2f}"
              f"{min_eig:>12.2e}{max_cond:>12.2e}{min_diag:>14.2e}")

    worst = state_dirs[-1] / "trained_hmm_params.npz"
    if worst.exists():
        params = np.load(worst, allow_pickle=True)
        covars = params["covars"]
        with open(state_dirs[-1] / "metadata.json", encoding="utf-8") as f:
            meta = json.load(f)
        if meta["covariance_type"] == "full":
            print(f"\nper-state covariance eigenvalue floor ({state_dirs[-1].name}):")
            for i in range(covars.shape[0]):
                eigvals = np.linalg.eigvalsh((covars[i] + covars[i].T) / 2)
                diag_vars = np.diag(covars[i])
                zero_dims = [feature_names[j] for j in range(len(diag_vars)) if diag_vars[j] < 1e-8]
                flag = f"  <-- near-zero var dims: {zero_dims}" if zero_dims else ""
                print(f"  state {i:>2}: min eig = {eigvals.min():.3e}, cond = {eigvals.max() / max(eigvals.min(), 1e-300):.2e}{flag}")


def main():
    args = parse_args()
    paths = paths_from_args(args)
    input_path = args.input_file or paths.hmm_data_dir / "hmm_data_train.npz"

    observations, feature_names = load_observations(input_path)
    diagnose_features(observations, feature_names)

    for exp_dir in args.exp_dir:
        diagnose_exp_dir(Path(exp_dir), observations.shape[0], list(feature_names))


if __name__ == "__main__":
    main()
