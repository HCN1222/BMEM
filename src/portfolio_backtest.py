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

# 全域交易參數
INITIAL_CAPITAL = 1000000 
FEE_BUY = 0.001425        
FEE_SELL = 0.001425       
TAX_SELL = 0.003          

# ==========================================
# 2. 定義 XGBoost 策略進場訊號
# ==========================================
print("\n2. 生成 XGBoost 策略進場訊號...")

# 策略: XGBoost (門檻 0.6)
XGB_THRESHOLD = 0.6
xgb_signals = df_eval[df_eval['pred_prob'] >= XGB_THRESHOLD].copy()
xgb_signals['date'] = xgb_signals['date'].astype(str).str[:10]
xgb_signals['sort_prob'] = xgb_signals['pred_prob'] # 換股依據: XGB 機率

print(f"-> XGBoost 策略共產生 {len(xgb_signals)} 個買進訊號")

# ==========================================
# 3. 建立 K 線資料庫與 0050 Benchmark
# ==========================================
print("\n3. 預載 K 線資料庫與 0050 大盤...")
all_signal_stocks = set(xgb_signals['stock_id'].unique())
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
# 4. 共用回測引擎函數 (修復高水位線與前視偏差，改為隔日開盤交易)
# ==========================================
all_dates = sorted(df_eval['date'].astype(str).str[:10].unique())

def run_top_n_backtest(signals_df, n, trailing_stop_ratio, hard_stop_ratio):
    cash = INITIAL_CAPITAL
    current_holdings = {}  # dict格式: {stock_id: info_dict}
    equity_curve = []   
    trade_history = []  

    # 💡 新增輔助函數：取得隔日開盤價與實際交易日期
    def get_next_open_price(stock_id, current_date):
        if stock_id in kline_cache:
            stock_dates = kline_cache[stock_id].index
            # 找出大於當前日期的所有未來交易日
            future_dates = stock_dates[stock_dates > current_date]
            if not future_dates.empty:
                next_date = future_dates[0] # 取下一個最近的交易日
                return kline_cache[stock_id].loc[next_date]['open'], next_date
        return None, None

    for today in all_dates:
        # A. 停損處理 (盤中觸發停損，維持原邏輯)
        stocks_to_remove = []
        for stock_id, holding in current_holdings.items():
            if stock_id in kline_cache and today in kline_cache[stock_id].index:
                row = kline_cache[stock_id].loc[today]
                
                if row['max'] > holding['highest_price']:
                    holding['highest_price'] = row['max']
                    
                stop_price_trailing = holding['highest_price'] * trailing_stop_ratio
                stop_price_cost = holding['buy_price'] * hard_stop_ratio
                stop_price = max(stop_price_trailing, stop_price_cost)
                
                if row['min'] <= stop_price:
                    sell_price = row['open'] if row['open'] < stop_price else stop_price
                    reason_str = '初始停損' if stop_price_cost >= stop_price_trailing else '移動停損'
                    
                    gross_proceeds = holding['shares'] * sell_price
                    net_proceeds = gross_proceeds * (1 - FEE_SELL - TAX_SELL) 
                    cash += net_proceeds
                    buy_cost = holding['shares'] * holding['buy_price'] * (1 + FEE_BUY)
                    
                    trade_history.append({
                        '股票代號': stock_id, '買進日期': holding['buy_date'], '賣出日期': today,
                        '報酬率': (net_proceeds / buy_cost) - 1, '備註': reason_str
                    })
                    stocks_to_remove.append(stock_id)
        
        for stock_id in stocks_to_remove:
            del current_holdings[stock_id]

        # B. 新進場與換股 (盤後決策，隔日開盤執行)
        today_signals = signals_df[signals_df['date'] == today]
        if not today_signals.empty:
            
            # 1. 更新現有持股的最高機率評分
            for stock_id, holding in current_holdings.items():
                if stock_id in today_signals['stock_id'].values:
                    new_prob = today_signals[today_signals['stock_id'] == stock_id].iloc[0]['sort_prob']
                    if new_prob > holding['buy_prob']:
                        holding['buy_prob'] = new_prob

            # 2. 找出新候選股
            candidates = today_signals[~today_signals['stock_id'].isin(current_holdings.keys())]
            candidates = candidates.sort_values('sort_prob', ascending=False)
            
            for _, candidate in candidates.iterrows():
                cand_id = candidate['stock_id']
                cand_prob = candidate['sort_prob']
                
                # 狀況一：投資組合還有空位，取得隔日開盤價買入
                if len(current_holdings) < n:
                    buy_price, actual_buy_date = get_next_open_price(cand_id, today)
                    
                    if buy_price is not None:
                        budget = cash / (n - len(current_holdings))
                        shares = int(budget / (buy_price * (1 + FEE_BUY)))
                        
                        if shares > 0:
                            cost = shares * buy_price * (1 + FEE_BUY)
                            cash -= cost
                            current_holdings[cand_id] = {
                                'stock_id': cand_id, 'shares': shares,
                                'buy_price': buy_price, 'highest_price': buy_price,
                                'buy_date': actual_buy_date, # 💡 紀錄為實際買入的隔天日期
                                'buy_prob': cand_prob
                            }
                            
                # 狀況二：投資組合已滿 n 檔，與最弱持股比較是否需要「汰弱留強」
                else:
                    weakest_stock_id = min(current_holdings, key=lambda k: current_holdings[k]['buy_prob'])
                    weakest_holding = current_holdings[weakest_stock_id]
                    
                    if cand_prob > weakest_holding['buy_prob']:
                        # 💡 決定換股：取得最弱持股的「隔日開盤價」賣出
                        sell_price, actual_sell_date = get_next_open_price(weakest_stock_id, today)
                        
                        # 備用機制：若隔天無交易資料 (例如下市)，用最後一天收盤價或成本價計算
                        if sell_price is None:
                            if weakest_stock_id in kline_cache and today in kline_cache[weakest_stock_id].index:
                                sell_price = kline_cache[weakest_stock_id].loc[today]['close']
                            else:
                                sell_price = weakest_holding['buy_price']
                            actual_sell_date = today
                            
                        gross_proceeds = weakest_holding['shares'] * sell_price
                        net_proceeds = gross_proceeds * (1 - FEE_SELL - TAX_SELL)
                        cash += net_proceeds
                        buy_cost = weakest_holding['shares'] * weakest_holding['buy_price'] * (1 + FEE_BUY)
                        
                        trade_history.append({
                            '股票代號': weakest_stock_id, '買進日期': weakest_holding['buy_date'], '賣出日期': actual_sell_date, # 💡 紀錄隔天日期
                            '報酬率': (net_proceeds / buy_cost) - 1, '備註': '換股'
                        })
                        del current_holdings[weakest_stock_id]
                        
                        # 💡 賣掉後，取得新候選股的「隔日開盤價」買入
                        buy_price, actual_buy_date = get_next_open_price(cand_id, today)
                        
                        if buy_price is not None:
                            budget = cash / (n - len(current_holdings))
                            shares = int(budget / (buy_price * (1 + FEE_BUY)))
                            
                            if shares > 0:
                                cost = shares * buy_price * (1 + FEE_BUY)
                                cash -= cost
                                current_holdings[cand_id] = {
                                    'stock_id': cand_id, 'shares': shares,
                                    'buy_price': buy_price, 'highest_price': buy_price,
                                    'buy_date': actual_buy_date, # 💡 紀錄為實際買入的隔天日期
                                    'buy_prob': cand_prob
                                }
                    else:
                        break # 若無法打敗最弱持股，停止審視後續較弱的候選股

        # C. 每日結算帳戶總淨值
        daily_equity = cash
        for stock_id, holding in current_holdings.items():
            if stock_id in kline_cache and today in kline_cache[stock_id].index:
                daily_equity += holding['shares'] * kline_cache[stock_id].loc[today]['close']
            else:
                daily_equity += holding['shares'] * holding['buy_price']
                
        equity_curve.append({'date': today, 'equity': daily_equity})

    # 期末強制平倉 (維持原狀)
    last_date = all_dates[-1]
    for stock_id, holding in list(current_holdings.items()):
        if stock_id in kline_cache and last_date in kline_cache[stock_id].index:
            sell_price = kline_cache[stock_id].loc[last_date]['close']
        else:
            sell_price = holding['buy_price']
            
        gross_proceeds = holding['shares'] * sell_price
        net_proceeds = gross_proceeds * (1 - FEE_SELL - TAX_SELL)
        buy_cost = holding['shares'] * holding['buy_price'] * (1 + FEE_BUY)
        
        trade_history.append({
            '股票代號': stock_id, '買進日期': holding['buy_date'], '賣出日期': last_date,
            '報酬率': (net_proceeds / buy_cost) - 1, '備註': '期末平倉'
        })
        
    equity_df = pd.DataFrame(equity_curve)
    equity_df['date'] = pd.to_datetime(equity_df['date'])
    equity_df['cummax'] = equity_df['equity'].cummax()
    equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax']
    
    mdd = equity_df['drawdown'].min()
    total_return = (equity_df.iloc[-1]['equity'] / INITIAL_CAPITAL) - 1
    
    return equity_df, trade_history, total_return, mdd

# ==========================================
# 5. 執行多組 n 的回測與繪圖
# ==========================================
print("\n4. 執行多檔持股 (Top-N) 回測對決...")
n_values = [1, 3, 5, 7, 9]
results = {}

for n in n_values:
    print(f"-> 正在回測 Top-{n} 策略...")
    # 維持原設定：初始停損 -20%, 移動停損 -10% 
    eq, trades, ret, mdd = run_top_n_backtest(xgb_signals, n, 0.80, 0.90)
    results[n] = {'equity': eq, 'trades': trades, 'ret': ret, 'mdd': mdd}

# 處理 0050 基準線
benchmark_return, benchmark_mdd = 0, 0
benchmark_dates = results[1]['equity']['date']  
df_0050_eq = pd.DataFrame({'date': benchmark_dates})

if df_0050 is not None:
    df_0050.index = pd.to_datetime(df_0050.index)
    benchmark_prices = df_0050.reindex(benchmark_dates, method='ffill')['close']
    benchmark_start_price = benchmark_prices.iloc[0]
    df_0050_eq['equity'] = (benchmark_prices.values / benchmark_start_price) * INITIAL_CAPITAL
    
    df_0050_eq['cummax'] = df_0050_eq['equity'].cummax()
    df_0050_eq['drawdown'] = (df_0050_eq['equity'] - df_0050_eq['cummax']) / df_0050_eq['cummax']
    benchmark_mdd = df_0050_eq['drawdown'].min()
    benchmark_return = (df_0050_eq['equity'].iloc[-1] / INITIAL_CAPITAL) - 1

print("\n5. 繪製並儲存比較圖表...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 11), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
colors = ['#d62728', '#ff7f0e', '#2ca02c', '#9467bd', '#e377c2'] 

# 繪製淨值曲線
for i, n in enumerate(n_values):
    res = results[n]
    ax1.plot(res['equity']['date'], res['equity']['equity'], color=colors[i], linewidth=2.0, 
             label=f'XGB Top-{n} (總報酬: {res["ret"]*100:.2f}%)')

if df_0050 is not None:
    ax1.plot(df_0050_eq['date'], df_0050_eq['equity'], color='#1f77b4', linewidth=1.5, linestyle='--', 
             alpha=0.8, label=f'0050 大盤 (總報酬: {benchmark_return*100:.2f}%)')

ax1.set_title('XGBoost 多檔持股 (Top-N) 策略大對決 vs 0050', fontsize=16, fontweight='bold')
ax1.set_ylabel('帳戶總淨值 (TWD)', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(loc='upper left', fontsize=11)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

# 繪製回撤圖
for i, n in enumerate(n_values):
    res = results[n]
    ax2.plot(res['equity']['date'], res['equity']['drawdown'] * 100, color=colors[i], linewidth=1.2, alpha=0.8,
             label=f'Top-{n} MDD: {res["mdd"]*100:.1f}%')

if df_0050 is not None:
    ax2.plot(df_0050_eq['date'], df_0050_eq['drawdown'] * 100, color='#1f77b4', linewidth=1.5, linestyle='--', 
             alpha=0.8, label=f'0050 MDD: {benchmark_mdd*100:.1f}%')
    
ax2.set_ylabel('回撤比例 (%)', fontsize=12)
ax2.set_xlabel('日期', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(loc='lower left', fontsize=10, ncol=2)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)

plt.tight_layout()
os.makedirs('./outputs/backtest', exist_ok=True)
save_path = './outputs/backtest/equity_curve_top_n_comparison.png'
plt.savefig(save_path, dpi=300)

# ==========================================
# 6. 輸出表格報告
# ==========================================
print("\n" + "="*85)
print(f" {'實盤模擬回測結果 (初始 100 萬) - 多檔持股大對決':^80} ")
print("="*85)

header = f"{'指標':<14} |"
for n in n_values:
    header += f" {'Top-'+str(n):<11} |"
header += f" {'0050大盤':<11}"
print(header)
print("-" * 85)

# 期末淨值
row_equity = f"{'期末淨值':<12} |"
for n in n_values:
    row_equity += f" {int(results[n]['equity'].iloc[-1]['equity']):<11,} |"
row_equity += f" {int(df_0050_eq['equity'].iloc[-1]) if df_0050 is not None else 'N/A':<11,}"
print(row_equity)

# 總報酬率
row_ret = f"{'總報酬率':<12} |"
for n in n_values:
    row_ret += f" {results[n]['ret']*100:>10.2f}% |"
row_ret += f" {benchmark_return*100:>10.2f}%"
print(row_ret)

# 最大回撤
row_mdd = f"{'最大回撤(MDD)':<12} |"
for n in n_values:
    row_mdd += f" {results[n]['mdd']*100:>10.2f}% |"
row_mdd += f" {benchmark_mdd*100:>10.2f}%"
print(row_mdd)

# 交易次數
row_trades = f"{'總交易次數':<11} |"
for n in n_values:
    row_trades += f" {len(results[n]['trades']):>8} 次 |"
row_trades += f" {'N/A':>8}"
print(row_trades)

# 勝率
row_win_rate = f"{'實戰勝率':<12} |"
for n in n_values:
    trades = results[n]['trades']
    win_r = sum(1 for t in trades if t['報酬率'] > 0) / len(trades) if len(trades)>0 else 0
    row_win_rate += f" {win_r*100:>10.2f}% |"
row_win_rate += f" {'N/A':>10}"
print(row_win_rate)

print("="*85)
print(f"✅ 圖表已成功儲存至: {save_path}")

plt.show()