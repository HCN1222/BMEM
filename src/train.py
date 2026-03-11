#!/usr/bin/env python3
"""
Train a Gaussian HMM using hmmlearn from a .npz dataset.

Expected .npz keys:
- lengths: 1D array of ints, shape (n_sequences,)
- observations: 2D array of floats, shape (total_timesteps, n_features)
- names: 1D array of strings, shape (n_sequences,)

Example usage:
python train_hmm.py \
    --input_file /path/to/data.npz \
    --outdir output \
    --iterations 50 \
    --n_states 3 \
    --covariance_type diag \
    --tol 1e-3 \
    --random_seed 42
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from hmmlearn.hmm import GaussianHMM
from tqdm import tqdm


def parse_args():
    """
    Parse command-line arguments.

    Common naming convention:
    - parse_args() returns an argparse.Namespace named 'args'
    """
    parser = argparse.ArgumentParser(
        description="Train a Gaussian HMM from concatenated multi-sequence observations stored in a .npz file."
    )

    parser.add_argument( "--input_file", type=str, required=True, help="Path to input .npz file." )
    parser.add_argument( "--outdir", type=str, required=True, help="Name or path of output directory. Default: output" )
    parser.add_argument( "--iterations", type=int, default=1000, help="Maximum number of EM iterations. Default: 50" )
    parser.add_argument( "--n_states", type=int, required=True, help="Number of hidden states." )
    parser.add_argument( "--covariance_type", type=str, default="full", choices=["full", "diag"], help='Covariance type for Gaussian emissions. Default: full')
    parser.add_argument( "--tol", type=float, default=1e-3, help="Tolerance for log-likelihood improvement to stop fitting. Recommended default: 1e-3" )
    parser.add_argument( "--random_seed", type=int, default=12, help="Random seed. Default: 12" )

    return parser.parse_args()


def validate_input_file(input_path: Path):
    """Check that the input file exists and has .npz suffix."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    if input_path.suffix.lower() != ".npz":
        raise ValueError(f"Input file must be a .npz file, got: {input_path.suffix}")


def load_dataset(npz_path: Path):
    """
    Load dataset from .npz file.

    Expected keys:
    - lengths
    - observations
    - names
    """
    data = np.load(npz_path, allow_pickle=True)

    required_keys = ["lengths", "observations", "names"]
    for key in required_keys:
        if key not in data:
            raise KeyError(f"Missing key '{key}' in input .npz file.")

    lengths = np.asarray(data["lengths"], dtype=int)
    observations = np.asarray(data["observations"], dtype=float)
    names = np.asarray(data["names"]).astype(str)

    if lengths.ndim != 1:
        raise ValueError(f"'lengths' must be 1D, got shape {lengths.shape}")

    if observations.ndim != 2:
        raise ValueError(
            f"'observations' must be 2D with shape (total_timesteps, n_features), got shape {observations.shape}"
        )

    if names.ndim != 1:
        raise ValueError(f"'names' must be 1D, got shape {names.shape}")

    if len(lengths) != len(names):
        raise ValueError(
            f"Number of sequences mismatch: len(lengths)={len(lengths)} but len(names)={len(names)}"
        )

    if np.any(lengths <= 0):
        raise ValueError("All sequence lengths must be positive integers.")

    if int(lengths.sum()) != observations.shape[0]:
        raise ValueError(
            f"Sum of lengths ({lengths.sum()}) does not match number of observation rows ({observations.shape[0]})."
        )

    return lengths, observations, names


def create_output_dir(input_path: Path, outdir_arg: str) -> Path:
    """
    Create an output directory with a timestamp under the given base path.
    Example:
        outdir_arg/result_20260311_142510
    """

    base_path = Path(outdir_arg)

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create folder name
    result_dir = base_path / f"result_{timestamp}"

    # Create directory
    result_dir.mkdir(parents=True, exist_ok=True)

    return result_dir


def train_model_with_progress(
    observations: np.ndarray,
    lengths: np.ndarray,
    n_states: int,
    covariance_type: str,
    max_iterations: int,
    tol: float,
    random_seed: int
):
    """
    Train GaussianHMM one EM step at a time so tqdm can display progress.

    Returns:
    - model
    - log_likelihood_history
    - converged_iteration (1-based index), or None if not early stopped
    """
    model = GaussianHMM( n_components=n_states, covariance_type=covariance_type,
        n_iter=1,              # one EM step per outer loop
        tol=tol,
        init_params="stmc",    # initialize only once
        params="stmc",
        random_state=random_seed,
        verbose=False
    )

    log_likelihood_history = []
    converged_iteration = None

    progress_bar = tqdm(range(max_iterations), desc="Training HMM", unit="iter")

    previous_loglik = None

    for iteration_idx in progress_bar:
        model.fit(observations, lengths=lengths)

        # After the first fit, prevent reinitialization
        model.init_params = ""

        current_loglik = model.score(observations, lengths=lengths)
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
    """
    Save log-likelihood curve after training.
    """
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
    names: np.ndarray,
    log_likelihood_history,
    args,
    lengths: np.ndarray,
    decoded_states: np.ndarray,
    posterior_probs: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray
):
    """
    Save trained parameters and metadata.
    """
    np.savez(
        outdir / "trained_hmm_params.npz",
        startprob=model.startprob_,
        transmat=model.transmat_,
        means=model.means_,
        covars=model.covars_,
        names=names,
        lengths=lengths,
        log_likelihood_history=np.asarray(log_likelihood_history, dtype=float),
        decoded_states=decoded_states,
        posterior_probabilities=posterior_probs,
        standardization_mean=feature_mean,
        standardization_std=feature_std
    )

    metadata = {
        "input_file": str(args.input_file),
        "outdir": str(outdir),
        "iterations": int(args.iterations),
        "n_states": int(args.n_states),
        "covariance_type": args.covariance_type,
        "tol": float(args.tol),
        "random_seed": int(args.random_seed),
        "n_sequences": int(len(names)),
        "n_total_timesteps": int(lengths.sum()),
        "n_features": int(model.means_.shape[1]),
        "trained_iterations": int(len(log_likelihood_history)),
        "final_log_likelihood": float(log_likelihood_history[-1]) if log_likelihood_history else None,
        "standardized_input": True
    }

    with open(outdir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def print_summary(names: np.ndarray, model: GaussianHMM):
    """
    Print sequence names and each learned Gaussian emission distribution.
    For a Gaussian HMM, the emission density distribution of each hidden state
    is defined by its mean vector and covariance matrix.
    """
    print("\nLoaded sequence names:")
    for i, name in enumerate(names):
        print(f"  [{i}] {name}")

    print("\nLearned Gaussian emission distributions by hidden state:")
    for state_idx in range(model.n_components):
        print(f"\nState {state_idx}")
        print("Mean:")
        print(model.means_[state_idx])
        print("Covariance:")
        print(model.covars_[state_idx])


def main():
    args = parse_args()

    input_path = Path(args.input_file)
    validate_input_file(input_path)

    outdir = create_output_dir(input_path, args.outdir)

    lengths, observations, names = load_dataset(input_path)
    n_features = observations.shape[1]

    print("Dataset loaded successfully.")
    print(f"Input file: {input_path}")
    print(f"Output directory: {outdir}")
    print(f"Number of sequences: {len(names)}")
    print(f"Total timesteps: {lengths.sum()}")
    print(f"Observation shape: {observations.shape}")
    print(f"Inferred number of observation features: {n_features}")
    print("Input observations will be standardized before training.")

    model, log_likelihood_history, converged_iteration = train_model_with_progress(
        observations=observations_std,
        lengths=lengths,
        n_states=args.n_states,
        covariance_type=args.covariance_type,
        max_iterations=args.iterations,
        tol=args.tol,
        random_seed=args.random_seed
    )

    decoded_states = model.predict(observations_std, lengths=lengths)
    posterior_probs = model.predict_proba(observations_std, lengths=lengths)

    save_results(
        outdir=outdir,
        model=model,
        names=names,
        log_likelihood_history=log_likelihood_history,
        args=args,
        lengths=lengths,
        decoded_states=decoded_states,
        posterior_probs=posterior_probs
    )

    save_loglik_plot(outdir, log_likelihood_history)

    print_summary(names, model)

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