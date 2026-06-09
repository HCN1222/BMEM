import pandas as pd
import numpy as np
import glob
import os
import sys
from hmmlearn import hmm
from pandas.api.indexers import FixedForwardWindowIndexer
from tqdm import tqdm

pd.set_option('display.unicode.east_asian_width', True)

# ==========================================
# 時間切點 (與設計文件一致)
# Train:  2021 ~ 2023-12-31
# Val:    2024-01-01 ~ 2024-12-31
# Test:   2025-01-01+  (封存回測集，從不參與訓練)
# ==========================================
TRAIN_END  = '2023-12-31'
VAL_START  = '2024-01-01'
VAL_END    = '2024-12-31'
TEST_START = '2025-01-01'

print("1. 正在載入資料與重建 HMM 模型...")
# ==========================================
# 1. 載入全部序列 (train + eval 合併，以取得所有日期的資料)
# ==========================================
train_parquet_path = './data/preprocessed_data/exp4/final_vectors_train.parquet'
eval_parquet_path  = './data/preprocessed_data/exp4/final_vectors_eval.parquet'
hmm_params_path    = './outputs/exp5/result_20260607_194920/states_6/trained_hmm_params.npz'

try:
    df_part1   = pd.read_parquet(train_parquet_path)
    df_part2   = pd.read_parquet(eval_parquet_path)
    hmm_params = np.load(hmm_params_path)
except Exception as e:
    print(f"檔案讀取失敗: {e}")
    sys.exit()

# 合併全部序列
df_all_raw = pd.concat([df_part1, df_part2], ignore_index=True)
df_all_raw['date'] = df_all_raw['date'].astype(str).str[:10]
print(f"合併後總資料筆數: {len(df_all_raw):,}")

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

print("2. 對全部序列進行滾動狀態機率計算 (Rolling Proba，防止未來函數)...")
# ==========================================
# 2. 全部序列使用 rolling_predict_proba，確保無未來函數
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

all_probs_list = []
grouped = df_all_raw.groupby('sequence_id', sort=False)
for seq_id, group in tqdm(grouped, desc="Rolling 機率", total=len(grouped)):
    p = rolling_predict_proba(group[feature_cols].values, model, window=120)
    all_probs_list.append(pd.DataFrame(p, index=group.index, columns=prob_cols))

prob_df_all = pd.concat(all_probs_list)
df_all_raw = pd.concat([df_all_raw, prob_df_all], axis=1)

print("3. 正在計算未來 2 週價格極值與實戰標籤...")
# ==========================================
# 3. 計算 Target Y (做多 / 做空)
# ==========================================
unique_stocks = df_all_raw['stock_id'].unique()
return_records = []
forward_indexer = FixedForwardWindowIndexer(window_size=10)

for stock_id in tqdm(unique_stocks, desc="計算未來價格極值"):
    _matches = glob.glob(f"./data/stocks/{stock_id}_2021-06-30_to_*.parquet")
    file_path = _matches[0] if _matches else ""
    if os.path.exists(file_path):
        kdf = pd.read_parquet(file_path)
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

print("4. 按日期切分並儲存三份資料集 (Train / Val / Test)...")
# ==========================================
# 4. 按時間切分：Train / Val / Test
# ==========================================
base_cols = [
    'date', 'stock_id', 'securities_trader_id', 'sequence_id',
    'net_buy_amt_60d', 'bias_60d'
] + feature_cols + prob_cols

df_train_period = df_all[df_all['date'] <= TRAIN_END].copy()
df_val_period   = df_all[(df_all['date'] >= VAL_START) & (df_all['date'] <= VAL_END)].copy()
df_test_period  = df_all[df_all['date'] >= TEST_START].copy()

print(f"\n切分結果:")
print(f"  Train (2021 ~ {TRAIN_END}): {len(df_train_period):,} 筆")
print(f"  Val   ({VAL_START} ~ {VAL_END}): {len(df_val_period):,} 筆")
print(f"  Test  ({TEST_START}+): {len(df_test_period):,} 筆")

os.makedirs('./data/preprocessed_data', exist_ok=True)

for mode, target_col in [('long', 'target_y_long'), ('short', 'target_y_short')]:
    df_tr = df_train_period[base_cols + [target_col]].rename(columns={target_col: 'target_y'})
    df_va = df_val_period[base_cols + [target_col]].rename(columns={target_col: 'target_y'})
    df_te = df_test_period[base_cols + [target_col]].rename(columns={target_col: 'target_y'})

    df_tr.to_parquet(f'./data/preprocessed_data/xgb_dataset_{mode}_train.parquet', index=False)
    df_va.to_parquet(f'./data/preprocessed_data/xgb_dataset_{mode}_val.parquet',   index=False)
    df_te.to_parquet(f'./data/preprocessed_data/xgb_dataset_{mode}_test.parquet',  index=False)

    print(f"\n[{mode}]")
    print(f"  Train: {len(df_tr):,} | Base rate: {df_tr['target_y'].mean()*100:.2f}%")
    print(f"  Val  : {len(df_va):,} | Base rate: {df_va['target_y'].mean()*100:.2f}%")
    print(f"  Test : {len(df_te):,} | Base rate: {df_te['target_y'].mean()*100:.2f}%")

print("\n完成！三份資料集已按正確時間切分儲存。")
