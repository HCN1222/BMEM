import argparse
import pandas as pd
import numpy as np
import sys
from hmmlearn import hmm
from pandas.api.indexers import FixedForwardWindowIndexer
from tqdm import tqdm  # 用於顯示進度條
from pathlib import Path

from src.utils.paths import add_broker_path_args, paths_from_args
from src.utils.stock_data import load_stock_data

# 開啟 Pandas 對齊全形中文字的支援
pd.set_option('display.unicode.east_asian_width', True)

parser = argparse.ArgumentParser(description='Evaluate a broker-specific HMM strategy.')
add_broker_path_args(parser)
parser.add_argument('--eval-parquet-path', type=Path, help='Override final_vectors_eval.parquet')
parser.add_argument('--hmm-params-path', type=Path, help='Override deployed HMM params')
parser.add_argument('--stock-info-dir', type=Path, help='Override shared stock data directory')
args = parser.parse_args()
paths = paths_from_args(args)

print("1. 正在載入 Eval 資料與重建 HMM 模型...")
# ==========================================
# 1. 載入資料與模型重建
# ==========================================
eval_parquet_path = args.eval_parquet_path or paths.hmm_data_dir / 'final_vectors_eval.parquet'
model_params_path = args.hmm_params_path or paths.hmm_model_path
stock_info_dir = args.stock_info_dir or paths.stock_dir

try:
    df_eval = pd.read_parquet(eval_parquet_path)
    hmm_params = np.load(model_params_path)
except Exception as e:
    print(f"檔案讀取失敗: {e}")
    sys.exit()

df_eval['date'] = df_eval['date'].astype(str).str[:10]

# 解析模型參數
startprob = hmm_params['startprob']
transmat = hmm_params['transmat']
means = hmm_params['means']
covars = hmm_params['covars']
feature_cols = ['z_t', 'c_t', 'a_t', 's_t', 'm_t']  # 確保與訓練時一致

n_components = means.shape[0]
n_features = means.shape[1]

# 自動判斷共變異數矩陣的形狀
if len(covars.shape) == 3:
    covar_type = "full"
elif len(covars.shape) == 2:
    covar_type = "diag"
else:
    covar_type = "spherical"

print(f"-> 偵測到 {n_components} 個隱藏狀態，Covariance Type: {covar_type}")

# ---------------------------------------------------------
# [修復區塊]：處理浮點數誤差，確保共變異數矩陣對稱且正定
# ---------------------------------------------------------
if covar_type == "full":
    for i in range(n_components):
        # 1. 強制對稱: (C + C.T) / 2
        covars[i] = (covars[i] + covars[i].T) / 2
        # 2. 強制正定: 在對角線加上一個微小值 (1e-5) 確保沒有極微小的負特徵值
        np.fill_diagonal(covars[i], covars[i].diagonal() + 1e-5)
elif covar_type == "diag":
    # 若為 diag，確保沒有小於等於 0 的變異數
    covars = np.maximum(covars, 1e-5)
# ---------------------------------------------------------

# 重建 HMM 模型
model = hmm.GaussianHMM(n_components=n_components, covariance_type=covar_type)
model.startprob_ = startprob
model.transmat_ = transmat
model.means_ = means
model.covars_ = covars  # 經過修復後，這裡就不會再報 ValueError 了！
model.n_features = n_features

print("2. 正在執行嚴格無未來函數的 Rolling Viterbi 解碼 (Window=120天)...")
# ==========================================
# 2. Rolling Viterbi Decoding (避免 Look-ahead Bias)
# ==========================================
def rolling_viterbi(sequence_features, hmm_model, window=120):
    """
    針對單一連續序列，每天只用歷史 window 天的資料來推論當天的 State
    """
    seq_len = len(sequence_features)
    states = np.zeros(seq_len, dtype=int)
    
    for t in range(seq_len):
        # 取出 max(0, t-120+1) 到 t 的特徵 (也就是包含 T 日在內的過去最多 120 天)
        start_idx = max(0, t - window + 1)
        X_window = sequence_features[start_idx : t + 1]
        
        # 進行解碼，並只取路徑的最後一天作為 T 日的真實狀態
        pred_path = hmm_model.predict(X_window)
        states[t] = pred_path[-1]
        
    return states

# [修復區塊]：棄用容易報錯的 progress_apply，改用標準 for 迴圈配合 tqdm
states_series_list = []
grouped = df_eval.groupby('sequence_id')

# 直接用 tqdm 包裝 groupby 物件來顯示進度條
for seq_id, group in tqdm(grouped, desc="推論進度", total=len(grouped)):
    # 針對每個 sequence 計算 rolling states
    pred_states = rolling_viterbi(group[feature_cols].values, model, window=120)
    
    # 轉換成 Series 並綁定原始的 index，確保後續對齊完全正確
    states_series_list.append(pd.Series(pred_states, index=group.index))

# 將所有計算好的 states 合併，並直接賦值回 df_eval
df_eval['State'] = pd.concat(states_series_list)
# ==========================================
# 3. 策略條件篩選 (EDA Filters)
# ==========================================
# 條件 1: 當日收盤價 < 60日成本價的 1.10 倍
cond_cost = (df_eval['cost_60d'] > 0) & (df_eval['close'] < df_eval['cost_60d'] * 1.10)

# 條件 2: 過去 60 天該券商累計買超金額 > 10 億 (依照你前一次分享的結果設定)
cond_amt = df_eval['net_buy_amt_60d'] > 1000000000

# 套用篩選
eda_df = df_eval[cond_cost & cond_amt].copy()

print(f"-> Eval Set 總筆數: {len(df_eval):,}")
print(f"-> 篩選後符合條件筆數: {len(eda_df):,}\n")

if len(eda_df) == 0:
    print("Eval 資料集中沒有任何資料符合此策略條件。")
    sys.exit()

print("4. 正在計算 Eval 樣本的「未來兩週綜合報酬」...")
# ==========================================
# 4. 針對篩選後的樣本計算未來 2 週報酬
# ==========================================
unique_stocks = eda_df['stock_id'].unique()
return_records = []
forward_indexer = FixedForwardWindowIndexer(window_size=10)

for stock_id in tqdm(unique_stocks, desc="計算未來報酬"):
    kdf = load_stock_data(stock_info_dir, stock_id)
    if not kdf.empty:
        kdf['date'] = kdf['date'].astype(str).str[:10]
        kdf = kdf.sort_values('date')
        
        kdf['future_2w_high'] = kdf['max'].shift(-1).rolling(window=forward_indexer, min_periods=1).max()
        kdf['future_2w_low']  = kdf['min'].shift(-1).rolling(window=forward_indexer, min_periods=1).min()
        kdf['future_2w_mean'] = kdf['close'].shift(-1).rolling(window=forward_indexer, min_periods=1).mean()
        
        kdf['high_ret'] = kdf['future_2w_high'] / kdf['close'] - 1
        kdf['low_ret']  = kdf['future_2w_low'] / kdf['close'] - 1
        kdf['mean_ret'] = kdf['future_2w_mean'] / kdf['close'] - 1
        
        temp_df = kdf[['date', 'high_ret', 'low_ret', 'mean_ret']].copy()
        temp_df['stock_id'] = str(stock_id)
        return_records.append(temp_df)

all_returns = pd.concat(return_records, ignore_index=True)
eda_df['stock_id'] = eda_df['stock_id'].astype(str)
eda_df = pd.merge(eda_df, all_returns, on=['stock_id', 'date'], how='left')

eda_df = eda_df.dropna(subset=['high_ret', 'low_ret', 'mean_ret'])

print("5. 產出 Out-of-Sample 策略勝率分析表...\n")
# ==========================================
# 5. 統計分析與報表輸出
# ==========================================
surge_threshold = 0.10  # 高點 >= 10%
drop_threshold = -0.10  # 低點 <= -10%

# 確保所有模型 states 都有顯示，即使部分 state 在 Eval 沒有觸發。
all_states = pd.DataFrame({'State': range(n_components)})
grouped = eda_df.groupby('State')

stats = pd.DataFrame()
stats['符合條件樣本數'] = grouped.size()
stats['正報酬機率(均值>0)'] = grouped['mean_ret'].apply(lambda x: (x > 0).mean())
stats['大漲機率(高點>10%)'] = grouped['high_ret'].apply(lambda x: (x >= surge_threshold).mean())
stats['大跌機率(低點<-10%)'] = grouped['low_ret'].apply(lambda x: (x <= drop_threshold).mean())
stats['平均高點報酬'] = grouped['high_ret'].mean()
stats['平均低點報酬'] = grouped['low_ret'].mean()
stats['高點中位數'] = grouped['high_ret'].median()
stats['低點中位數'] = grouped['low_ret'].median()

# 將所有存在的 state 合併回去，處理某些 state 在 Eval 完全沒觸發的狀況
stats = all_states.merge(stats, on='State', how='left').fillna(0)

format_cols = [col for col in stats.columns if col not in ['State', '符合條件樣本數']]
for col in format_cols:
    stats[col] = (stats[col] * 100).round(2).astype(str) + '%'
stats['符合條件樣本數'] = stats['符合條件樣本數'].astype(int)

print("="*105)
print("     [Eval OOS 測試] 策略: 股價 < 60日成本1.1倍 且 60日買超 > 10億 (無未來函數 Rolling Viterbi)")
print("="*105)
print(stats.to_markdown(index=False))
