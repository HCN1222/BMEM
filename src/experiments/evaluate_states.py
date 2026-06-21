#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 直接從你的 train.py 匯入所有精華函數
from src.experiments.train_hmm import (
    load_dataset, 
    train_model_with_progress, 
    create_output_dir, 
    save_results, 
    save_loglik_plot, 
    print_summary
)
from src.utils.paths import add_broker_path_args, paths_from_args

def calculate_bic(model, X, lengths, covariance_type):
    """計算 BIC (包含 full 與 diag 的自動判斷)"""
    n_features = X.shape[1]
    n_states = model.n_components
    
    # 計算自由參數數量 (k)
    if covariance_type == "full":
        cov_params = n_states * (n_features * (n_features + 1)) / 2
    else: # diag
        cov_params = n_states * n_features
        
    n_params = (n_states * (n_states - 1) + 
                (n_states - 1) + 
                n_states * n_features + 
                cov_params)
    
    log_likelihood = model.score(X, lengths)
    n_samples = X.shape[0]
    
    # BIC = -2 * ln(L) + k * ln(N)
    bic = -2 * log_likelihood + n_params * np.log(n_samples)
    return bic, log_likelihood, n_params

def main():
    parser = argparse.ArgumentParser(description="Evaluate HMM states, save ALL models, and plot BIC.")
    add_broker_path_args(parser)
    parser.add_argument("--input-file", "--input_file", type=Path, help="Override the broker HMM training .npz file.")
    parser.add_argument("--outdir", type=Path, help="Override the broker HMM evaluation runs directory.")
    parser.add_argument("--iterations", type=int, default=200, help="Max EM iterations.")
    parser.add_argument("--covariance_type", type=str, default="full", choices=["full", "diag"])
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--random_seed", type=int, default=25)
    parser.add_argument("--min_states", type=int, default=2, help="Minimum number of states to evaluate.")
    parser.add_argument("--max_states", type=int, default=6, help="Maximum number of states to evaluate.")
    
    args = parser.parse_args()
    paths = paths_from_args(args)
    input_path = args.input_file or paths.hmm_data_dir / "hmm_data_train.npz"
    output_root = args.outdir or paths.hmm_runs_dir
    args.input_file = str(input_path)
    
    # 1. 載入資料
    lengths, observations, feature_names = load_dataset(input_path)
    print(f"Data loaded. Observations: {observations.shape[0]}, Features: {observations.shape[1]}")
    
    # 2. 建立「主評估資料夾」(帶有唯一時間戳記，避免覆蓋)
    eval_master_dir = create_output_dir(input_path, output_root)
    print(f"\nCreated master evaluation directory: {eval_master_dir}")
    
    bics = []
    log_likelihoods = []
    state_range = range(args.min_states, args.max_states + 1)
    
    best_bic = np.inf
    best_n = None

    print("\n" + "="*60)
    print(" STARTING MODEL TRAINING & EVALUATION")
    print("="*60)
    
    # 3. 跑迴圈評估不同 State，並「逐一存檔」
    for n_states in state_range:
        print(f"\n--- Training & Saving model with n_states = {n_states} ---")
        
        # 訓練模型
        model, history, converged_iteration = train_model_with_progress(
            observations=observations,
            lengths=lengths,
            n_states=n_states,
            covariance_type=args.covariance_type,
            max_iterations=args.iterations,
            tol=args.tol,
            random_seed=args.random_seed
        )
        
        # 計算 BIC
        bic, log_l, params = calculate_bic(model, observations, lengths, args.covariance_type)
        bics.append(bic)
        log_likelihoods.append(log_l)
        
        print(f"-> Log-Lik: {log_l:.2f} | Params: {params} | BIC: {bic:.2f}")
        
        if bic < best_bic:
            best_bic = bic
            best_n = n_states
            
        # 為這個 State 建立獨立的子資料夾 (例如: states_3)
        state_dir = eval_master_dir / f"states_{n_states}"
        state_dir.mkdir(parents=True, exist_ok=True)
        
        # 進行 Viterbi Decode 預測狀態
        decoded_states = model.predict(observations, lengths=lengths)
        posterior_probs = model.predict_proba(observations, lengths=lengths)
        
        # 動態更新 args.n_states，確保 metadata.json 存到正確的數字
        args.n_states = n_states
        
        # 將該 State 的所有結果存入專屬子資料夾
        save_results(
            outdir=state_dir,
            model=model,
            feature_names=feature_names,
            log_likelihood_history=history,
            args=args,
            lengths=lengths,
            decoded_states=decoded_states,
            posterior_probs=posterior_probs
        )
        save_loglik_plot(state_dir, history)
        print(f"-> Successfully saved to: {state_dir}")

    # ==========================================
    # 4. 生成並儲存總結的 BIC 圖表 (存在主資料夾)
    # ==========================================
    plt.figure(figsize=(10, 6))
    plt.plot(state_range, bics, marker='o', linestyle='-', color='b', label='BIC')
    plt.axvline(best_n, color='r', linestyle='--', label=f'Best n_states = {best_n}')
    plt.title('HMM State Selection (BIC)')
    plt.xlabel('Number of Hidden States')
    plt.ylabel('BIC Score (Lower is better)')
    plt.xticks(state_range)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    # 存檔至主資料夾
    bic_plot_path = eval_master_dir / "bic_evaluation_curve.png"
    plt.savefig(bic_plot_path, dpi=150)
    plt.close()

    print("\n" + "="*60)
    print(f"All models saved successfully!")
    print(f"Master Directory: {eval_master_dir}")
    print(f"BIC Chart saved at: {bic_plot_path}")
    print(f"Optimal States Selected (Lowest BIC): {best_n}")
    print("="*60)

if __name__ == "__main__":
    main()
