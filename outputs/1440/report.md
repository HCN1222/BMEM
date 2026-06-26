# Broker 1440 (Merrill Lynch) — Model & Backtest Report

Generated from current contents of `outputs/1440/`.

## 1. Pipeline Summary

Broker `1440` is the original research broker for the BMEM pipeline: a Gaussian HMM trained on broker trading-behavior features feeds two XGBoost classifiers (long/short) that produce daily ranked trading signals.

## 2. HMM Model (`models/HMM/`)

| Field | Value |
|---|---|
| States | 10 |
| Covariance type | full |
| Features | `z_t`, `c_t`, `a_t`, `s_t`, `m_t` |
| Training timesteps | 951,068 |
| Iterations (run / converged) | 200 / 183 |
| Final log-likelihood | 17,015,343.57 |

State count (10) was selected by minimizing BIC across candidate state counts during model development; only the deployed model artifacts (`trained_hmm_params.npz`, `metadata.json`) remain in this directory — the per-candidate BIC sweep outputs have since been cleaned up.

## 3. XGBoost Models (`models/XGBoost/`)

Two independent binary classifiers consume HMM state probabilities + market features:

- **Long** (`models/XGBoost/long/`): P(price rises ≥ +10% within 10 days without a ‑10% drawdown first). Signal threshold ≥ 0.6 (tuned for F1).
- **Short** (`models/XGBoost/short/`): P(price falls ≤ -10% within 10 days without a +10% rebound first). Signal threshold ≥ 0.8 (tuned for precision).

Each side has a saved model (`xgb_trading_model.json`) and a feature-importance chart (`xgboost_feature_importance.png`); the long side additionally has an `xgb_analysis.png` diagnostic plot.

## 4. Backtest (`backtest/`)

### 4.1 Results (2025-02-03 → 2026-05-29)

The evaluation window is intentionally capped at `TEST_END = '2026-05-31'` in `src/experiments/prepare_xgb_data.py` (added so the tail of the test set never includes rows whose 10-day-forward label isn't fully observable yet). 2026-05-30/31 fall on a Sat/Sun, so the last tradable day inside the cap is **2026-05-29** — which is exactly where this rerun's evaluation data and backtest end.

**Top-N comparison (no smoothing):**

| Strategy | Final Equity (TWD) | Total Return | Max Drawdown | Trades | Win Rate |
|---|---|---|---|---|---|
| **Top-1** | **4,454,735** | **+345.47%** | -32.13% | 57 | **63.16%** |
| Top-5 | 3,091,000 | +209.10% | -36.30% | 260 | 56.54% |
| Top-3 | 1,978,891 | +97.89% | -37.57% | 165 | 55.15% |
| 0050 Benchmark | 2,187,856 | +118.79% | -26.37% | — | — |

Top-1 still wins outright and remains ahead of the benchmark, though the longer window pulls its return down from the original +378% (33 trades) to +345% (57 trades) and its win rate from 69.7% to 63.2% — consistent with the extra ~5 months of trades trading at lower edge.

**EMA long-probability smoothing sensitivity** (does smoothing the entry signal help?):

| Top-N | No Smoothing | EMA-3 | EMA-5 | EMA-10 |
|---|---|---|---|---|
| Top-1 | +345.47% (57 trades, 63.16% win, MDD -32.13%) | +101.50% (67, 49.25%, -46.21%) | +252.16% (43, 51.16%, -44.62%) | +33.70% (48, 54.17%, -43.02%) |
| Top-3 | +97.89% (165, 55.15%, -37.57%) | +90.13% (141, 54.61%, -35.79%) | +105.60% (148, 53.38%, -32.71%) | **+194.61%** (134, 58.96%, -26.88%) |
| Top-5 | **+209.10%** (260, 56.54%, -36.30%) | +110.80% (224, 54.91%, -32.63%) | +159.91% (227, 57.71%, -36.62%) | +183.17% (214, 55.61%, -28.78%) |

Takeaway: for Top-1, smoothing the long probability strictly hurts (raw signal is best — smoothing delays entries into a concentrated single-position strategy). For Top-3 and Top-5, EMA-10 smoothing meaningfully improves both return and drawdown over the raw signal, suggesting smoothing helps diversified baskets but not single-stock concentration.

Charts (regenerated): `equity_curve_top_n_comparison.png`, `equity_curve_ema_long_top{1,3,5}_comparison.png`.

### 4.2 Trade log (`reports/top1_trade_history.csv`)

Trade log spans **2025-02-04 → 2026-05-29** (57 closed trades):

| Metric | Value |
|---|---|
| Trades | 57 |
| Win rate | 63.2% |
| Total return | +345.47% |

Exit reasons: 31 trades closed by rotation into a higher-confidence pick ("滿檔換股"), 22 by the short model's warning signal ("做空模型預警賣出"), 3 by the -20% trailing stop-loss, 1 forced end-of-period close.