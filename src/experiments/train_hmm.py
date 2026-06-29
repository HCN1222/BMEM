#!/usr/bin/env python3
"""
Train a Gaussian HMM using hmmlearn from a .npz dataset with K-Means initialization.
Assumes input observations are already preprocessed/standardized.

Expected .npz keys:
- lengths: 1D array of ints, shape (n_sequences,)
- observations: 2D array of floats, shape (total_timesteps, n_features)
- feature_names: 1D array of strings
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.cluster import KMeans
from tqdm import tqdm

from src.utils.paths import add_broker_path_args, paths_from_args


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Gaussian HMM from concatenated multi-sequence observations stored in a .npz file."
    )

    add_broker_path_args(parser)
    parser.add_argument( "--input-file", "--input_file", type=Path, help="Override the broker HMM training .npz file." )
    parser.add_argument( "--outdir", type=Path, help="Override the deployed HMM model directory." )
    parser.add_argument( "--iterations", type=int, default=100, help="Maximum number of EM iterations. Default: 100" )
    parser.add_argument( "--n_states", type=int, required=True, help="Number of hidden states." )
    parser.add_argument( "--covariance_type", type=str, default="full", choices=["full", "diag"], help='Covariance type for Gaussian emissions. Default: full')
    parser.add_argument( "--tol", type=float, default=1e-3, help="Tolerance for log-likelihood improvement to stop fitting. Recommended default: 1e-3" )
    parser.add_argument( "--random_seed", type=int, default=12, help="Random seed. Default: 12" )
    parser.add_argument( "--n_restarts", type=int, default=1, help="Number of random restarts. Best log-likelihood is kept. Default: 1" )

    return parser.parse_args()


def validate_input_file(input_path: Path):
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")
    if input_path.suffix.lower() != ".npz":
        raise ValueError(f"Input file must be a .npz file, got: {input_path.suffix}")


def load_dataset(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)

    required_keys = ["lengths", "observations", "feature_names"]
    for key in required_keys:
        if key not in data:
            raise KeyError(f"Missing key '{key}' in input .npz file.")

    lengths = np.asarray(data["lengths"], dtype=int)
    observations = np.asarray(data["observations"], dtype=float)
    feature_names = np.asarray(data["feature_names"]).astype(str)

    if lengths.ndim != 1:
        raise ValueError(f"'lengths' must be 1D, got shape {lengths.shape}")
    if observations.ndim != 2:
        raise ValueError(f"'observations' must be 2D with shape (total_timesteps, n_features), got shape {observations.shape}")
    if feature_names.ndim != 1:
        raise ValueError(f"'feature_names' must be 1D, got shape {feature_names.shape}")
    if lengths.sum() != len(observations):
        raise ValueError(f"Number of sequences mismatch: sum(lengths)={sum(lengths)} but len(observations)={len(observations)}")
    if np.any(lengths <= 0):
        raise ValueError("All sequence lengths must be positive integers.")

    return lengths, observations, feature_names


def create_output_dir(input_path: Path, outdir_arg: str) -> Path:
    base_path = Path(outdir_arg)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = base_path / f"result_{timestamp}"
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir


def initialize_hmm_with_kmeans(observations: np.ndarray, n_states: int, covariance_type: str, random_seed: int):
    print("Running K-Means for parameter initialization...")
    kmeans = KMeans(n_clusters=n_states, random_state=random_seed, n_init="auto")
    labels = kmeans.fit_predict(observations)
    
    n_features = observations.shape[1]
    means = kmeans.cluster_centers_
    
    if covariance_type == "full":
        covars = np.zeros((n_states, n_features, n_features))
        for i in range(n_states):
            obs_i = observations[labels == i]
            if len(obs_i) > 1:
                covars[i] = np.cov(obs_i, rowvar=False) + np.eye(n_features) * 1e-3
            else:
                covars[i] = np.eye(n_features)
    elif covariance_type == "diag":
        covars = np.zeros((n_states, n_features))
        for i in range(n_states):
            obs_i = observations[labels == i]
            if len(obs_i) > 1:
                covars[i] = np.var(obs_i, axis=0) + 1e-6
            else:
                covars[i] = np.ones(n_features)

    startprob = np.full(n_states, 1.0 / n_states)
    transmat = np.full((n_states, n_states), 1.0 / n_states)

    return startprob, transmat, means, covars


def _repair_covars(model: GaussianHMM, covariance_type: str, n_states: int, epsilon: float = 1e-3):
    """Project each covariance to nearest positive-definite matrix via eigendecomposition."""
    if covariance_type == "full":
        n_features = model.means_.shape[1]
        for i in range(n_states):
            cov = model.covars_[i]
            if not np.all(np.isfinite(cov)):
                model.covars_[i] = np.eye(n_features) * epsilon
                continue
            cov = (cov + cov.T) / 2
            try:
                eigvals, eigvecs = np.linalg.eigh(cov)
                eigvals = np.maximum(eigvals, epsilon)
                model.covars_[i] = eigvecs @ np.diag(eigvals) @ eigvecs.T
            except np.linalg.LinAlgError:
                model.covars_[i] = np.eye(n_features) * epsilon
    elif covariance_type == "diag":
        model.covars_ = np.maximum(model.covars_, epsilon)


def train_model_with_progress(
    observations: np.ndarray,
    lengths: np.ndarray,
    n_states: int,
    covariance_type: str,
    max_iterations: int,
    tol: float,
    random_seed: int
):
    startprob, transmat, means, covars = initialize_hmm_with_kmeans(
        observations, n_states, covariance_type, random_seed
    )

    model = GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        min_covar=1e-3,
        n_iter=1,
        tol=tol,
        init_params="",
        params="stmc",
        random_state=random_seed,
        verbose=False
    )

    model.startprob_ = startprob
    model.transmat_ = transmat
    model.means_ = means
    model.covars_ = covars

    # ==========================================
    # STICKY HMM QUICK FIX: Set Transition Prior
    # ==========================================
    stickiness_weight = 500.0 #1000.0  # Adjust this! Higher = more sticky
    transmat_prior = np.ones((n_states, n_states))
    np.fill_diagonal(transmat_prior, stickiness_weight)
    model.transmat_prior = transmat_prior
    # ==========================================

    log_likelihood_history = []
    converged_iteration = None
    progress_bar = tqdm(range(max_iterations), desc="Training HMM", unit="iter")
    previous_loglik = None

    for iteration_idx in progress_bar:
        try:
            model.fit(observations, lengths=lengths)
            current_loglik = model.score(observations, lengths=lengths)
        except (np.linalg.LinAlgError, ValueError) as e:
            print(f"\n  WARNING: 第 {iteration_idx + 1} 次迭代 covariance 矩陣數值不穩定 ({e})，嘗試修復...")
            repaired = False
            for attempt, eps in enumerate([1e-3, 1e-2, 1e-1], start=1):
                try:
                    _repair_covars(model, covariance_type, n_states, epsilon=eps)
                    model.fit(observations, lengths=lengths)
                    current_loglik = model.score(observations, lengths=lengths)
                    print(f"  -> 修復成功 (epsilon={eps}，第 {attempt} 次嘗試)")
                    repaired = True
                    break
                except (np.linalg.LinAlgError, ValueError):
                    continue
            if not repaired:
                print("  -> 修復失敗，跳過此次迭代")
                current_loglik = previous_loglik if previous_loglik is not None else -float("inf")

        log_likelihood_history.append(float(current_loglik))

        if previous_loglik is None:
            progress_bar.set_postfix(loglik=f"{current_loglik:.4f}", improvement="N/A")
        else:
            improvement = current_loglik - previous_loglik
            progress_bar.set_postfix(
                loglik=f"{current_loglik:.4f}",
                improvement=f"{improvement:.6f}"
            )

            if abs(improvement) < tol:
                converged_iteration = iteration_idx + 1
                break

        previous_loglik = current_loglik

    return model, log_likelihood_history, converged_iteration


def save_loglik_plot(outdir: Path, log_likelihood_history):
    if not log_likelihood_history:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(log_likelihood_history) + 1), log_likelihood_history, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Log-likelihood")
    plt.title("HMM Training Log-likelihood")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outdir / "log_likelihood_curve.png", dpi=150)
    plt.close()


def save_results(
    outdir: Path,
    model: GaussianHMM,
    feature_names: np.ndarray,
    log_likelihood_history,
    args,
    lengths: np.ndarray,
    decoded_states: np.ndarray,
    posterior_probs: np.ndarray
):
    np.savez(
        outdir / "trained_hmm_params.npz",
        startprob=model.startprob_,
        transmat=model.transmat_,
        means=model.means_,
        covars=model.covars_,
        lengths=lengths,
        log_likelihood_history=np.asarray(log_likelihood_history, dtype=float),
        decoded_states=decoded_states,
        posterior_probabilities=posterior_probs
    )

    metadata = {
        "broker_id": getattr(args, "broker_id", None),
        "input_file": str(args.input_file),
        "outdir": str(outdir),
        "iterations": int(args.iterations),
        "n_states": int(args.n_states),
        "covariance_type": args.covariance_type,
        "tol": float(args.tol),
        "random_seed": int(args.random_seed),
        "n_total_timesteps": int(lengths.sum()),
        "feature_names": feature_names.tolist(),
        "n_features": int(model.means_.shape[1]),
        "trained_iterations": int(len(log_likelihood_history)),
        "final_log_likelihood": float(log_likelihood_history[-1]) if log_likelihood_history else None
    }

    with open(outdir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def print_summary(feature_names: np.ndarray, model: GaussianHMM):
    print("\nLearned Gaussian emission distributions by hidden state:")
    print("features:", feature_names)
    for state_idx in range(model.n_components):
        print(f"\nState {state_idx}")
        print("Mean:")
        print(model.means_[state_idx])
        print("Covariance:")
        print(model.covars_[state_idx])


def main():
    args = parse_args()
    paths = paths_from_args(args)

    input_path = args.input_file or paths.hmm_data_dir / "hmm_data_train.npz"
    validate_input_file(input_path)

    outdir = args.outdir or paths.hmm_model_dir
    outdir.mkdir(parents=True, exist_ok=True)
    args.input_file = str(input_path)
    args.outdir = str(outdir)

    lengths, observations, feature_names = load_dataset(input_path)

    print("Dataset loaded successfully.")
    print(f"Input file: {input_path}")
    print(f"Output directory: {outdir}")
    print(f"Total timesteps: {lengths.sum()}")
    print(f"Observation shape: {observations.shape}")
    print("features:", feature_names)

    best_model, best_history, best_converged = None, None, None
    best_loglik = -float("inf")
    best_seed = args.random_seed
    for restart in range(args.n_restarts):
        seed = args.random_seed + restart
        print(f"\nRestart {restart + 1}/{args.n_restarts} (seed={seed})")
        m, h, c = train_model_with_progress(
            observations=observations,
            lengths=lengths,
            n_states=args.n_states,
            covariance_type=args.covariance_type,
            max_iterations=args.iterations,
            tol=args.tol,
            random_seed=seed
        )
        loglik = h[-1] if h else -float("inf")
        if loglik > best_loglik:
            best_loglik = loglik
            best_seed = seed
            best_model, best_history, best_converged = m, h, c
            print(f"-> New best log-lik: {best_loglik:.2f} (seed={seed})")
    model, log_likelihood_history, converged_iteration = best_model, best_history, best_converged
    print(f"\nBest seed: {best_seed} | Final log-lik: {best_loglik:.2f}")

    decoded_states = model.predict(observations, lengths=lengths)
    posterior_probs = model.predict_proba(observations, lengths=lengths)

    save_results(
        outdir=outdir,
        model=model,
        feature_names=feature_names,
        log_likelihood_history=log_likelihood_history,
        args=args,
        lengths=lengths,
        decoded_states=decoded_states,
        posterior_probs=posterior_probs
    )

    save_loglik_plot(outdir, log_likelihood_history)
    print_summary(feature_names, model)

    print("\nTraining completed.")
    print(f"Total training iterations run: {len(log_likelihood_history)}")
    if converged_iteration is not None:
        print(f"Early stopping triggered at iteration: {converged_iteration}")
    else:
        print("Reached the maximum number of iterations without early stopping.")

    if log_likelihood_history:
        print(f"Final log-likelihood: {log_likelihood_history[-1]:.6f}")
        print(f"Saved log-likelihood plot to: {outdir / 'log_likelihood_curve.png'}")
    print(f"Saved decoded states and posterior probabilities to: {outdir / 'trained_hmm_params.npz'}")


if __name__ == "__main__":
    main()
