import argparse
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from hmmlearn import hmm
from pandas.api.indexers import FixedForwardWindowIndexer
from tqdm import tqdm

from src.utils.paths import add_broker_path_args, paths_from_args
from src.utils.stock_data import load_stock_data

pd.set_option('display.unicode.east_asian_width', True)

# ==========================================
# 切分規則
# Evaluation : 依照 preprocess.py 算出的 is_eval 真實斷點 (每支股票自己的斷點，不是統一日曆)，
#              並另外用 TEST_END 封頂，避免納入還沒有足夠未來報酬可算標籤的尾端資料。
# Validation : 從 Train (~is_eval) 裡面再切最後 25%，切點對齊最近的連續假日斷點，
#              確保不會把同一支股票的同一段連續資料切成兩半。
# ==========================================
TEST_END = '2026-05-31'
VAL_FRACTION = 0.25

parser = argparse.ArgumentParser(description="Prepare broker-specific XGBoost datasets.")
add_broker_path_args(parser)
parser.add_argument('--train-parquet-path', type=Path, help='Override final_vectors_train.parquet')
parser.add_argument('--eval-parquet-path', type=Path, help='Override final_vectors_eval.parquet')
parser.add_argument('--hmm-params-path', type=Path, help='Override the deployed HMM params')
parser.add_argument('--stock-info-dir', type=Path, help='Override the shared stock data directory')
parser.add_argument('--output-dir', type=Path, help='Override the broker XGBoost dataset directory')
args = parser.parse_args()
paths = paths_from_args(args)


def find_holiday_cutoff(dates: pd.Series, val_fraction: float, search_radius_days: int = 60) -> pd.Timestamp:
    """
    Pick a Train/Val cutoff date such that roughly `val_fraction` of the rows in
    `dates` fall after it, then snap that cutoff to the nearest multi-day market
    holiday gap (the biggest gap-in-calendar-days within a search window around
    the target point) so no stock's continuous sequence gets split mid-stream.

    Returns the last date that belongs to Train (Val starts the following
    trading day).
    """
    dates = pd.to_datetime(dates)
    counts_by_date = dates.value_counts().sort_index()
    cum_frac = counts_by_date.cumsum() / counts_by_date.sum()
    target_date = cum_frac[cum_frac >= (1 - val_fraction)].index[0]

    unique_dates = counts_by_date.index.to_numpy()
    # gap_days[i] = calendar days between unique_dates[i] and unique_dates[i-1]
    gap_days = np.diff(unique_dates) / np.timedelta64(1, 'D')
    gap_after_date = unique_dates[1:]  # the gap BEFORE this date

    window_mask = (
        (gap_after_date >= target_date - pd.Timedelta(days=search_radius_days)) &
        (gap_after_date <= target_date + pd.Timedelta(days=search_radius_days))
    )
    if not window_mask.any():
        return pd.Timestamp(target_date)

    candidate_dates = gap_after_date[window_mask]
    candidate_gaps = gap_days[window_mask]
    best_idx = np.argmax(candidate_gaps)
    holiday_start_date = candidate_dates[best_idx]

    cutoff_pos = np.where(unique_dates == holiday_start_date)[0][0] - 1
    return pd.Timestamp(unique_dates[cutoff_pos])


print("1. 正在載入 Train / Eval 資料與重建 HMM 模型...")
# ==========================================
# 1. Train 與 Eval 分開載入，不在這個階段合併
# ==========================================
train_parquet_path = args.train_parquet_path or paths.hmm_data_dir / 'final_vectors_train.parquet'
eval_parquet_path  = args.eval_parquet_path or paths.hmm_data_dir / 'final_vectors_eval.parquet'
hmm_params_path    = args.hmm_params_path or paths.hmm_model_path
stock_info_dir     = args.stock_info_dir or paths.stock_dir
output_dir         = args.output_dir or paths.xgboost_data_dir()

try:
    df_train   = pd.read_parquet(train_parquet_path)
    df_eval    = pd.read_parquet(eval_parquet_path)
    hmm_params = np.load(hmm_params_path)
except Exception as e:
    print(f"檔案讀取失敗: {e}")
    sys.exit()

df_train['date'] = df_train['date'].astype(str).str[:10]
df_eval['date']  = df_eval['date'].astype(str).str[:10]
print(f"Train 筆數: {len(df_train):,} | Eval 筆數: {len(df_eval):,}")

feature_cols = ['z_t', 'c_t', 'a_t', 's_t', 'm_t']

# 解析模型參數並重建 HMM 模型
startprob    = hmm_params['startprob']
transmat     = hmm_params['transmat']
means        = hmm_params['means']
covars       = hmm_params['covars']
n_components = means.shape[0]

if len(covars.shape) == 3:
    covar_type = "full"
elif len(covars.shape) == 2:
    covar_type = "diag"
else:
    covar_type = "spherical"

if covar_type == "full":
    for i in range(n_components):
        covars[i] = (covars[i] + covars[i].T) / 2
        np.fill_diagonal(covars[i], covars[i].diagonal() + 1e-5)
elif covar_type == "diag":
    covars = np.maximum(covars, 1e-5)

model = hmm.GaussianHMM(n_components=n_components, covariance_type=covar_type)
model.startprob_ = startprob
model.transmat_  = transmat
model.means_     = means
model.covars_    = covars
model.n_features = len(feature_cols)

prob_cols = [f'prob_S{i}' for i in range(n_components)]
print(f"HMM 重建完成: {n_components} states, covariance={covar_type}")

print("2a. Train 機率直接重用 HMM 訓練時的 posterior_probabilities...")
# ==========================================
# 2a. Train: 不重算，直接用訓練當下 Baum-Welch 算好的後驗機率
# ==========================================
posterior_probs_train = hmm_params['posterior_probabilities']
if len(df_train) != len(posterior_probs_train):
    min_len = min(len(df_train), len(posterior_probs_train))
    print(f"  WARNING: 長度不一致 (df_train={len(df_train):,}, posterior={len(posterior_probs_train):,})，"
          f"取尾端對齊 {min_len:,} 筆")
    df_train = df_train.iloc[-min_len:].reset_index(drop=True)
    posterior_probs_train = posterior_probs_train[-min_len:]
else:
    df_train = df_train.reset_index(drop=True)

prob_df_train = pd.DataFrame(posterior_probs_train, columns=prob_cols)
df_train = pd.concat([df_train, prob_df_train], axis=1)

print("2b. Eval 機率使用 Rolling Proba (120日視窗，防止未來函數)...")
# ==========================================
# 2b. Eval: 只對 Eval 序列做 rolling_predict_proba，確保無未來函數
# ==========================================
def rolling_predict_proba(sequence_features, hmm_model, window=120):
    seq_len = len(sequence_features)
    probs = np.zeros((seq_len, n_components))
    for t in range(seq_len):
        start_idx = max(0, t - window + 1)
        X_window = sequence_features[start_idx : t + 1]
        window_probs = hmm_model.predict_proba(X_window)
        probs[t] = window_probs[-1]
    return probs

eval_probs_list = []
grouped_eval = df_eval.groupby('sequence_id', sort=False)
for seq_id, group in tqdm(grouped_eval, desc="Eval Rolling 機率", total=len(grouped_eval)):
    p = rolling_predict_proba(group[feature_cols].values, model, window=120)
    eval_probs_list.append(pd.DataFrame(p, index=group.index, columns=prob_cols))

prob_df_eval = pd.concat(eval_probs_list)
df_eval = pd.concat([df_eval, prob_df_eval], axis=1)

print("3. 正在計算未來 2 週價格極值與實戰標籤...")
# ==========================================
# 3. 計算 Target Y (做多 / 做空)，合併僅為了批次算報酬標籤，is_eval 標記保留
# ==========================================
df_train['is_eval'] = False
df_eval['is_eval']  = True
df_all_raw = pd.concat([df_train, df_eval], ignore_index=True)

unique_stocks = df_all_raw['stock_id'].unique()
return_records = []
forward_indexer = FixedForwardWindowIndexer(window_size=10)

for stock_id in tqdm(unique_stocks, desc="計算未來價格極值"):
    kdf = load_stock_data(stock_info_dir, stock_id)
    if not kdf.empty:
        kdf['date'] = kdf['date'].astype(str).str[:10]
        kdf = kdf.sort_values('date')

        if str(stock_id) == '0050':
            split_mask = kdf['date'] >= '2025-06-18'
            kdf.loc[split_mask, ['close', 'max', 'min']] *= 4

        kdf['future_2w_high'] = kdf['max'].shift(-1).rolling(window=forward_indexer, min_periods=1).max()
        kdf['future_2w_low']  = kdf['min'].shift(-1).rolling(window=forward_indexer, min_periods=1).min()

        kdf['high_ret'] = kdf['future_2w_high'] / kdf['close'] - 1
        kdf['low_ret']  = kdf['future_2w_low'] / kdf['close'] - 1

        temp_df = kdf[['date', 'high_ret', 'low_ret']].copy()
        temp_df['stock_id'] = str(stock_id)
        return_records.append(temp_df)

all_returns = pd.concat(return_records, ignore_index=True)
df_all_raw['stock_id'] = df_all_raw['stock_id'].astype(str)
df_all = pd.merge(df_all_raw, all_returns, on=['stock_id', 'date'], how='left')
df_all = df_all.dropna(subset=['high_ret', 'low_ret'])

df_all['target_y_long']  = ((df_all['high_ret'] >= 0.10) & (df_all['low_ret'] > -0.10)).astype(int)
df_all['target_y_short'] = ((df_all['low_ret'] <= -0.10) & (df_all['high_ret'] < 0.10)).astype(int)

print("4. 切分 Train / Validation / Evaluation...")
# ==========================================
# 4. Evaluation = is_eval 真實斷點 (封頂 TEST_END)
#    Validation  = 從 Train (~is_eval) 裡再切最後 25%，切點對齊最近的連續假日
# ==========================================
df_test_period = df_all[df_all['is_eval'] & (df_all['date'] <= TEST_END)].copy()
df_train_all   = df_all[~df_all['is_eval']].copy()

cutoff_date = find_holiday_cutoff(df_train_all['date'], val_fraction=VAL_FRACTION)
print(f"  Train/Val 切點 (對齊最近連續假日): {cutoff_date.date()}")

train_dates = pd.to_datetime(df_train_all['date'])
df_train_period = df_train_all[train_dates <= cutoff_date].copy()
df_val_period   = df_train_all[train_dates >  cutoff_date].copy()

print(f"\n切分結果:")
print(f"  Train      (~ {cutoff_date.date()}): {len(df_train_period):,} 筆")
print(f"  Validation ({cutoff_date.date()} 之後): {len(df_val_period):,} 筆")
print(f"  Evaluation (is_eval, ~ {TEST_END}): {len(df_test_period):,} 筆")

base_cols = [
    'date', 'stock_id', 'securities_trader_id', 'sequence_id',
    'net_buy_amt_60d', 'bias_60d'
] + feature_cols + prob_cols

for mode, target_col in [('long', 'target_y_long'), ('short', 'target_y_short')]:
    df_tr = df_train_period[base_cols + [target_col]].rename(columns={target_col: 'target_y'})
    df_va = df_val_period[base_cols + [target_col]].rename(columns={target_col: 'target_y'})
    df_te = df_test_period[base_cols + [target_col]].rename(columns={target_col: 'target_y'})

    side_dir = output_dir / mode
    side_dir.mkdir(parents=True, exist_ok=True)
    df_tr.to_parquet(side_dir / 'train.parquet', index=False)
    df_va.to_parquet(side_dir / 'validation.parquet', index=False)
    df_te.to_parquet(side_dir / 'evaluation.parquet', index=False)

    print(f"\n[{mode}]")
    print(f"  Train      : {len(df_tr):,} | Base rate: {df_tr['target_y'].mean()*100:.2f}%")
    print(f"  Validation : {len(df_va):,} | Base rate: {df_va['target_y'].mean()*100:.2f}%")
    print(f"  Evaluation : {len(df_te):,} | Base rate: {df_te['target_y'].mean()*100:.2f}%")

print("\n完成！Evaluation 依 is_eval 真實斷點切分，Validation 從 Train 切 25% 並對齊最近連續假日。")
