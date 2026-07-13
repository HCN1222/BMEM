#!/usr/bin/env python3
import argparse
import json
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
    print_summary,
    _repair_covars,
)
from src.utils.paths import add_broker_path_args, paths_from_args

def bic_from_metadata(metadata_path: Path, n_samples: int):
    """從已存的 metadata.json 重新計算 BIC，用於 resume 時跳過已完成的 state。"""
    with open(metadata_path, encoding="utf-8") as f:
        meta = json.load(f)
    n_states = meta["n_states"]
    n_features = meta["n_features"]
    covariance_type = meta["covariance_type"]
    log_likelihood = meta["final_log_likelihood"]
    cov_params = n_states * n_features if covariance_type == "diag" else n_states * (n_features * (n_features + 1)) / 2
    n_params = n_states * (n_states - 1) + (n_states - 1) + n_states * n_features + cov_params
    bic = -2 * log_likelihood + n_params * np.log(n_samples)
    return bic, log_likelihood

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

def save_seed_selection_plot(outdir: Path, seed_log):
    """畫出每個 state 數最終採用的 seed（星號）以及各 restart seed 的 log-likelihood 分布。
    只涵蓋本次實際訓練的 state（resume 跳過的不含在內）。"""
    from matplotlib.lines import Line2D

    entries = [e for e in seed_log if np.isfinite(e.get("best_log_likelihood", -np.inf))]
    if not entries:
        return None

    max_restarts = max(max((len(e.get("restarts", [])) for e in entries), default=1), 1)
    cmap = plt.cm.viridis

    plt.figure(figsize=(11, 6))
    for e in entries:
        n = e["n_states"]
        for r in e.get("restarts", []):
            ll = r.get("log_likelihood")
            if ll is None or not np.isfinite(ll):
                continue
            frac = r.get("restart", 0) / max(max_restarts - 1, 1)
            plt.scatter(n, ll, color=cmap(frac), s=45, alpha=0.75, zorder=2)
        plt.scatter(n, e["best_log_likelihood"], marker="*", s=280,
                    facecolors="gold", edgecolors="black", zorder=3)
        plt.annotate(f"seed={e['best_seed']}", (n, e["best_log_likelihood"]),
                     textcoords="offset points", xytext=(0, 11), ha="center", fontsize=8)

    handles = [Line2D([], [], marker="o", linestyle="", color=cmap(j / max(max_restarts - 1, 1)),
                      label=f"restart {j}") for j in range(max_restarts)]
    handles.append(Line2D([], [], marker="*", linestyle="", markerfacecolor="gold",
                          markeredgecolor="black", markersize=15, label="chosen (best log-lik)"))
    plt.legend(handles=handles, loc="best", fontsize=8)
    plt.title("Seed Selection per State Count (best restart starred)")
    plt.xlabel("Number of Hidden States")
    plt.ylabel("Final Log-likelihood")
    plt.xticks([e["n_states"] for e in entries])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    path = outdir / "seed_selection.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path

def main():
    parser = argparse.ArgumentParser(description="Evaluate HMM states, save ALL models, and plot BIC.")
    add_broker_path_args(parser)
    parser.add_argument("--input-file", "--input_file", type=Path, help="Override the broker HMM training .npz file.")
    parser.add_argument("--outdir", type=Path, help="Override the broker HMM evaluation runs directory.")
    parser.add_argument("--iterations", type=int, default=200, help="Max EM iterations.")
    parser.add_argument("--covariance_type", type=str, default="full", choices=["full", "diag"])
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--random_seed", type=int, default=25)
    parser.add_argument("--n_restarts", type=int, default=1, help="Number of random restarts per state count. Best log-likelihood is kept.")
    parser.add_argument("--min_states", type=int, default=2, help="Minimum number of states to evaluate.")
    parser.add_argument("--max_states", type=int, default=6, help="Maximum number of states to evaluate.")
    parser.add_argument("--resume-dir", "--resume_dir", type=Path, default=None, help="Resume an interrupted run from this existing directory. Completed states are skipped.")

    args = parser.parse_args()
    paths = paths_from_args(args)
    input_path = args.input_file or paths.hmm_data_dir / "hmm_data_train.npz"
    output_root = args.outdir or paths.hmm_runs_dir
    args.input_file = str(input_path)

    # 1. 載入資料
    lengths, observations, feature_names = load_dataset(input_path)
    print(f"Data loaded. Observations: {observations.shape[0]}, Features: {observations.shape[1]}")

    # 2. 決定輸出資料夾：resume 模式直接用既有資料夾，否則建新的
    if args.resume_dir:
        eval_master_dir = Path(args.resume_dir).resolve()
        if not eval_master_dir.exists():
            raise FileNotFoundError(f"Resume directory not found: {eval_master_dir}")
        print(f"\nResuming from existing directory: {eval_master_dir}")
    else:
        eval_master_dir = create_output_dir(input_path, output_root)
        print(f"\nCreated master evaluation directory: {eval_master_dir}")
    
    bics = []
    log_likelihoods = []
    # Resume 時載入既有 seed_log，避免覆蓋掉先前已訓練 state 的 seed 紀錄
    seed_log = []
    if args.resume_dir:
        existing_seed_log = eval_master_dir / "seed_log.json"
        if existing_seed_log.exists():
            try:
                seed_log = json.load(open(existing_seed_log, encoding="utf-8"))
                print(f"Loaded existing seed_log (states: {sorted(e['n_states'] for e in seed_log)})")
            except Exception:
                seed_log = []
    state_range = range(args.min_states, args.max_states + 1)

    best_bic = np.inf
    best_n = None

    print("\n" + "="*60)
    print(" STARTING MODEL TRAINING & EVALUATION")
    print("="*60)
    
    # 3. 跑迴圈評估不同 State，並「逐一存檔」
    for n_states in state_range:
        state_dir = eval_master_dir / f"states_{n_states}"
        meta_path = state_dir / "metadata.json"

        # Resume 模式：已完成的 state 直接從 metadata 讀取，跳過訓練
        if meta_path.exists():
            bic, log_l = bic_from_metadata(meta_path, observations.shape[0])
            print(f"\n--- states={n_states} already done, skipping. Log-Lik: {log_l:.2f} | BIC: {bic:.2f} ---")
            bics.append(bic)
            log_likelihoods.append(log_l)
            if bic < best_bic:
                best_bic = bic
                best_n = n_states
            continue

        print(f"\n--- Training & Saving model with n_states = {n_states} (restarts={args.n_restarts}) ---")

        # 多次 restart，取 log-likelihood 最高的模型
        best_model, best_history, best_converged = None, None, None
        best_restart_loglik = -np.inf
        best_seed = args.random_seed
        restart_records = []
        for restart in range(args.n_restarts):
            seed = args.random_seed + restart
            print(f"  Restart {restart + 1}/{args.n_restarts} (seed={seed})")
            m, h, c = train_model_with_progress(
                observations=observations,
                lengths=lengths,
                n_states=n_states,
                covariance_type=args.covariance_type,
                max_iterations=args.iterations,
                tol=args.tol,
                random_seed=seed
            )
            restart_loglik = h[-1] if h else -np.inf
            restart_records.append({"restart": restart, "seed": seed, "log_likelihood": float(restart_loglik)})
            if restart_loglik > best_restart_loglik:
                best_restart_loglik = restart_loglik
                best_seed = seed
                best_model, best_history, best_converged = m, h, c
                print(f"  -> New best log-lik: {best_restart_loglik:.2f}")
        model, history, converged_iteration = best_model, best_history, best_converged
        print(f"  => Best seed for n_states={n_states}: seed={best_seed} (log-lik={best_restart_loglik:.2f})")
        seed_log.append({
            "n_states": n_states,
            "best_seed": best_seed,
            "best_log_likelihood": float(best_restart_loglik),
            "restarts": restart_records,
        })

        # 計算 BIC：直接用訓練得到的 log-likelihood（與 resume 的 bic_from_metadata 完全一致）。
        # 不再對模型重新 score——先前的 _repair_covars(1e-2) 會對「未標準化、變異數極小」的特徵
        # 套過大的絕對 floor，汙染 covariance 導致 loglik 崩壞、BIC 錯誤。
        log_l = best_restart_loglik
        n_features = observations.shape[1]
        cov_params = (n_states * n_features if args.covariance_type == "diag"
                      else n_states * (n_features * (n_features + 1)) / 2)
        params = n_states * (n_states - 1) + (n_states - 1) + n_states * n_features + cov_params
        bic = -2 * log_l + params * np.log(observations.shape[0])
        bics.append(bic)
        log_likelihoods.append(log_l)
        
        print(f"-> Log-Lik: {log_l:.2f} | Params: {params} | BIC: {bic:.2f}")
        
        if bic < best_bic:
            best_bic = bic
            best_n = n_states
            
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

    seed_log_path = eval_master_dir / "seed_log.json"
    # 依 n_states 去重（保留最後訓練的）並排序，避免 resume 重疊產生重複項
    seed_log = sorted({e["n_states"]: e for e in seed_log}.values(), key=lambda e: e["n_states"])
    with open(seed_log_path, "w", encoding="utf-8") as f:
        json.dump(seed_log, f, indent=2, ensure_ascii=False)

    seed_plot_path = save_seed_selection_plot(eval_master_dir, seed_log)

    print("\n" + "="*60)
    print(f"All models saved successfully!")
    print(f"Master Directory: {eval_master_dir}")
    print(f"BIC Chart saved at: {bic_plot_path}")
    print(f"Seed log saved at: {seed_log_path}")
    if seed_plot_path is not None:
        print(f"Seed selection chart saved at: {seed_plot_path}")
    print(f"Optimal States Selected (Lowest BIC): {best_n}")
    print("="*60)

if __name__ == "__main__":
    main()
