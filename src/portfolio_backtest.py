import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import sys
from tqdm import tqdm

# 設定中文字體與負號顯示
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False

print("1. 載入 XGBoost 模型與 Eval 測試資料...")
# ==========================================
# 1. 載入模型與準備資料
# ==========================================
eval_path = './data/preprocessed_data/xgb_dataset_eval.parquet'
model_path = './outputs/models/xgb_trading_model.json'

try:
    df_eval = pd.read_parquet(eval_path)
    clf = xgb.XGBClassifier()
    clf.load_model(model_path)
except Exception as e:
    print(f"檔案讀取失敗: {e}")
    sys.exit()

prob_cols = [f'prob_S{i}' for i in range(10)]
feature_cols = ['z_t', 'c_t', 'a_t', 's_t', 'm_t', 'bias_60d', 'net_buy_amt_60d'] + prob_cols

# XGBoost 預測
X_eval = df_eval[feature_cols]
df_eval['pred_prob'] = clf.predict_proba(X_eval)[:, 1]

# 找出每日的 HMM 離散狀態 (機率最高者)
df_eval['current_state'] = df_eval[prob_cols].idxmax(axis=1)

# 全域交易參數
INITIAL_CAPITAL = 1000000 
FEE_BUY = 0.001425        
FEE_SELL = 0.001425       
TAX_SELL = 0.003          

# ==========================================
# 2. 定義兩個策略的訊號與停損參數
# ==========================================
print("\n2. 生成 XGBoost 與 EDA 策略進場訊號...")

# 策略 A: XGBoost (門檻 0.70)
XGB_THRESHOLD = 0.6
xgb_signals = df_eval[df_eval['pred_prob'] >= XGB_THRESHOLD].copy()
xgb_signals['date'] = xgb_signals['date'].astype(str).str[:10]
xgb_signals['sort_prob'] = xgb_signals['pred_prob'] # 換股依據: XGB 機率

# 策略 B: EDA 專家策略 (60日買超 > 10億 + 乖離 < 10% + State 2)
eda_signals = df_eval[
    (df_eval['net_buy_amt_60d'] > 1000000000) &
    (df_eval['bias_60d'] < 0.10) &
    (df_eval['current_state'] == 'prob_S2')
].copy()
eda_signals['date'] = eda_signals['date'].astype(str).str[:10]
eda_signals['sort_prob'] = eda_signals['prob_S2'] # 換股依據: S2 機率

print(f"-> XGBoost 策略共產生 {len(xgb_signals)} 個買進訊號")
print(f"-> EDA 專家策略共產生 {len(eda_signals)} 個買進訊號")

# ==========================================
# 3. 建立 K 線資料庫與 0050 Benchmark
# ==========================================
print("\n3. 預載 K 線資料庫與 0050 大盤...")
all_signal_stocks = set(xgb_signals['stock_id'].unique()).union(set(eda_signals['stock_id'].unique()))
kline_cache = {}

for stock_id in tqdm(all_signal_stocks, desc="載入個股 K 線"):
    kline_path = f"./data/stocks/{stock_id}_2021-06-30_to_2026-02-11.parquet"
    if os.path.exists(kline_path):
        kdf = pd.read_parquet(kline_path)
        kdf['date'] = kdf['date'].astype(str).str[:10]
        kdf = kdf.sort_values('date').set_index('date')
        kline_cache[stock_id] = kdf

df_0050 = None
benchmark_path = "./data/stocks/0050_2021-06-30_to_2026-02-11.parquet"
if os.path.exists(benchmark_path):
    df_0050 = pd.read_parquet(benchmark_path)
    df_0050['date'] = df_0050['date'].astype(str).str[:10]
    df_0050 = df_0050.sort_values('date').set_index('date')
    
    SPLIT_DATE = '2025-06-18' 
    if SPLIT_DATE in df_0050.index or df_0050.index.max() >= SPLIT_DATE:
        df_0050.loc[df_0050.index >= SPLIT_DATE, 'close'] *= 4
    print("✅ 成功載入 0050 作為大盤基準線。")

# ==========================================
# 4. 共用回測引擎函數
# ==========================================
all_dates = sorted(df_eval['date'].astype(str).str[:10].unique())

def run_single_all_in_backtest(signals_df, trailing_stop_ratio, hard_stop_ratio):
    cash = INITIAL_CAPITAL
    current_holding = None  
    equity_curve = []   
    trade_history = []  

    for today in all_dates:
        # A. 停損處理
        if current_holding is not None:
            stock_id = current_holding['stock_id']
            if stock_id in kline_cache and today in kline_cache[stock_id].index:
                row = kline_cache[stock_id].loc[today]
                
                if row['max'] > current_holding['highest_price']:
                    current_holding['highest_price'] = row['max']
                    
                stop_price_trailing = current_holding['highest_price'] * trailing_stop_ratio
                stop_price_cost = current_holding['buy_price'] * hard_stop_ratio
                stop_price = max(stop_price_trailing, stop_price_cost)
                
                if row['min'] <= stop_price:
                    sell_price = row['open'] if row['open'] < stop_price else stop_price
                    reason_str = '初始停損' if stop_price_cost >= stop_price_trailing else '移動停損'
                    
                    gross_proceeds = current_holding['shares'] * sell_price
                    net_proceeds = gross_proceeds * (1 - FEE_SELL - TAX_SELL) 
                    cash += net_proceeds
                    buy_cost = current_holding['shares'] * current_holding['buy_price'] * (1 + FEE_BUY)
                    
                    trade_history.append({
                        '股票代號': stock_id, '買進日期': current_holding['buy_date'], '賣出日期': today,
                        '報酬率': (net_proceeds / buy_cost) - 1, '備註': reason_str
                    })
                    current_holding = None

        # B. 新進場與換股
        today_signals = signals_df[signals_df['date'] == today]
        if not today_signals.empty:
            today_signals = today_signals.sort_values('sort_prob', ascending=False)
            best_signal = today_signals.iloc[0]
            best_stock_id = best_signal['stock_id']
            best_prob = best_signal['sort_prob']
            
            if current_holding is not None:
                if best_prob > current_holding['buy_prob'] and best_stock_id != current_holding['stock_id']:
                    stock_id = current_holding['stock_id']
                    if stock_id in kline_cache and today in kline_cache[stock_id].index:
                        sell_price = kline_cache[stock_id].loc[today]['close']
                    else:
                        sell_price = current_holding['buy_price']
                        
                    gross_proceeds = current_holding['shares'] * sell_price
                    net_proceeds = gross_proceeds * (1 - FEE_SELL - TAX_SELL)
                    cash += net_proceeds
                    buy_cost = current_holding['shares'] * current_holding['buy_price'] * (1 + FEE_BUY)
                    
                    trade_history.append({
                        '股票代號': stock_id, '買進日期': current_holding['buy_date'], '賣出日期': today,
                        '報酬率': (net_proceeds / buy_cost) - 1, '備註': '換股'
                    })
                    current_holding = None
                elif best_prob > current_holding['buy_prob'] and best_stock_id == current_holding['stock_id']:
                    current_holding['buy_prob'] = best_prob

            if current_holding is None:
                if best_stock_id in kline_cache and today in kline_cache[best_stock_id].index:
                    buy_price = kline_cache[best_stock_id].loc[today]['close']
                    shares = int(cash / (buy_price * (1 + FEE_BUY)))
                    if shares > 0:
                        cash -= shares * buy_price * (1 + FEE_BUY)
                        current_holding = {
                            'stock_id': best_stock_id, 'shares': shares,
                            'buy_price': buy_price, 'highest_price': buy_price,
                            'buy_date': today, 'buy_prob': best_prob  
                        }

        # C. 每日結算
        daily_equity = cash
        if current_holding is not None:
            stock_id = current_holding['stock_id']
            if stock_id in kline_cache and today in kline_cache[stock_id].index:
                daily_equity += current_holding['shares'] * kline_cache[stock_id].loc[today]['close']
            else:
                daily_equity += current_holding['shares'] * current_holding['buy_price']
                
        equity_curve.append({'date': today, 'equity': daily_equity})

    # 期末強制平倉
    last_date = all_dates[-1]
    if current_holding is not None:
        stock_id = current_holding['stock_id']
        if stock_id in kline_cache and last_date in kline_cache[stock_id].index:
            sell_price = kline_cache[stock_id].loc[last_date]['close']
        else:
            sell_price = current_holding['buy_price']
            
        gross_proceeds = current_holding['shares'] * sell_price
        net_proceeds = gross_proceeds * (1 - FEE_SELL - TAX_SELL)
        buy_cost = current_holding['shares'] * current_holding['buy_price'] * (1 + FEE_BUY)
        
        trade_history.append({
            '股票代號': stock_id, '買進日期': current_holding['buy_date'], '賣出日期': last_date,
            '報酬率': (net_proceeds / buy_cost) - 1, '備註': '期末平倉'
        })
        
    # 計算 MDD
    equity_df = pd.DataFrame(equity_curve)
    equity_df['date'] = pd.to_datetime(equity_df['date'])
    equity_df['cummax'] = equity_df['equity'].cummax()
    equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax']
    
    mdd = equity_df['drawdown'].min()
    total_return = (equity_df.iloc[-1]['equity'] / INITIAL_CAPITAL) - 1
    
    return equity_df, trade_history, total_return, mdd

print("\n4. 執行 XGBoost 策略與 EDA 策略實盤回測...")
# 執行 XGBoost (初始停損 -10%, 移動停損 -20%)
xgb_eq, xgb_trades, xgb_ret, xgb_mdd = run_single_all_in_backtest(xgb_signals, 0.80, 0.90)

# 執行 EDA 專家策略 (依你的要求: 無初始停損限制，僅最高點 -10% 移動停損)
eda_eq, eda_trades, eda_ret, eda_mdd = run_single_all_in_backtest(eda_signals, 0.90, 0.00)

# 處理 0050 基準線
benchmark_return, benchmark_mdd = 0, 0
if df_0050 is not None:
    df_0050.index = pd.to_datetime(df_0050.index)
    benchmark_prices = df_0050.reindex(xgb_eq['date'], method='ffill')['close']
    benchmark_start_price = benchmark_prices.iloc[0]
    xgb_eq['0050_equity'] = (benchmark_prices.values / benchmark_start_price) * INITIAL_CAPITAL
    
    xgb_eq['0050_cummax'] = xgb_eq['0050_equity'].cummax()
    xgb_eq['0050_drawdown'] = (xgb_eq['0050_equity'] - xgb_eq['0050_cummax']) / xgb_eq['0050_cummax']
    benchmark_mdd = xgb_eq['0050_drawdown'].min()
    benchmark_return = (xgb_eq['0050_equity'].iloc[-1] / INITIAL_CAPITAL) - 1

print("\n5. 繪製並儲存終極大對決比較圖...")
# ==========================================
# 5. 畫圖與報表輸出
# ==========================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

# 淨值曲線
ax1.plot(xgb_eq['date'], xgb_eq['equity'], color='#d62728', linewidth=2.5, label=f'XGBoost 機器學習 (總報酬: {xgb_ret*100:.2f}%)')
ax1.plot(eda_eq['date'], eda_eq['equity'], color='#2ca02c', linewidth=2.0, linestyle='--', label=f'EDA 專家策略 (總報酬: {eda_ret*100:.2f}%)')
if df_0050 is not None:
    ax1.plot(xgb_eq['date'], xgb_eq['0050_equity'], color='#1f77b4', linewidth=1.5, alpha=0.8, label=f'0050 台灣50 (總報酬: {benchmark_return*100:.2f}%)')

ax1.set_title('量化策略終極對決：XGBoost vs EDA基礎策略 vs 0050大盤', fontsize=16, fontweight='bold')
ax1.set_ylabel('帳戶總淨值 (TWD)', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(loc='upper left', fontsize=11)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

# 回撤圖
ax2.plot(xgb_eq['date'], xgb_eq['drawdown'] * 100, color='#d62728', linewidth=1.5, label=f'XGB 回撤 (MDD: {xgb_mdd*100:.1f}%)')
ax2.plot(eda_eq['date'], eda_eq['drawdown'] * 100, color='#2ca02c', linewidth=1.5, linestyle='--', label=f'EDA 回撤 (MDD: {eda_mdd*100:.1f}%)')
if df_0050 is not None:
    ax2.plot(xgb_eq['date'], xgb_eq['0050_drawdown'] * 100, color='#1f77b4', linewidth=1.2, alpha=0.7, label=f'0050 回撤 (MDD: {benchmark_mdd*100:.1f}%)')
    
ax2.set_ylabel('回撤比例 (%)', fontsize=12)
ax2.set_xlabel('日期', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(loc='lower left', fontsize=10)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)

plt.tight_layout()
os.makedirs('./outputs/backtest', exist_ok=True)
plt.savefig('./outputs/backtest/equity_curve_ultimate_showdown.png', dpi=300)

print("="*70)
print("             實盤模擬回測大對決 (初始 100 萬)")
print("="*70)
print(f"{'指標':<15} | {'XGBoost 策略':<15} | {'EDA 基礎策略':<15} | {'0050 大盤':<15}")
print("-" * 70)
print(f"{'期末淨值':<15} | {int(xgb_eq.iloc[-1]['equity']):<15,} | {int(eda_eq.iloc[-1]['equity']):<15,} | {int(xgb_eq['0050_equity'].iloc[-1]) if df_0050 is not None else 'N/A':<15,}")
print(f"{'總報酬率':<15} | {xgb_ret*100:>14.2f}% | {eda_ret*100:>14.2f}% | {benchmark_return*100:>14.2f}%")
print(f"{'最大回撤 (MDD)':<13} | {xgb_mdd*100:>14.2f}% | {eda_mdd*100:>14.2f}% | {benchmark_mdd*100:>14.2f}%")

if len(xgb_trades) > 0 and len(eda_trades) > 0:
    xgb_win_rate = sum(1 for t in xgb_trades if t['報酬率'] > 0) / len(xgb_trades)
    eda_win_rate = sum(1 for t in eda_trades if t['報酬率'] > 0) / len(eda_trades)
    print(f"{'總交易次數':<15} | {len(xgb_trades):>14} 次 | {len(eda_trades):>14} 次 | {'N/A':>15}")
    print(f"{'實戰勝率':<15} | {xgb_win_rate*100:>14.2f}% | {eda_win_rate*100:>14.2f}% | {'N/A':>15}")
print("="*70)
print("✅ 終極大對決圖表已儲存至: ./outputs/backtest/equity_curve_ultimate_showdown.png")

plt.show()