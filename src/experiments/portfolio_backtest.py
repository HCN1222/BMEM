import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys
from pathlib import Path
from tqdm import tqdm

from src.utils.paths import add_broker_path_args, paths_from_args
from src.utils.stock_data import load_stock_data

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

parser = argparse.ArgumentParser(description='Run a broker-specific portfolio backtest.')
add_broker_path_args(parser)
parser.add_argument('--eval-path', type=Path, help='Override the long-side XGBoost test dataset')
parser.add_argument('--model-long-path', type=Path, help='Override the long XGBoost model')
parser.add_argument('--model-short-path', type=Path, help='Override the short XGBoost model')
parser.add_argument('--stock-info-dir', type=Path, help='Override the shared stock data directory')
parser.add_argument('--output-dir', type=Path, help='Override the broker backtest directory')
args = parser.parse_args()
paths = paths_from_args(args)

print("1. 載入 XGBoost 雙模型與 Evaluation 資料...")
# ==========================================
# 1. 載入模型與準備資料
# ==========================================
eval_path = args.eval_path or paths.xgboost_data_dir('long') / 'evaluation.parquet'
model_long_path = args.model_long_path or paths.xgboost_model_path('long')
model_short_path = args.model_short_path or paths.xgboost_model_path('short')
stock_info_dir = args.stock_info_dir or paths.stock_dir
output_dir = args.output_dir or paths.backtest_dir
output_dir.mkdir(parents=True, exist_ok=True)

try:
    df_eval = pd.read_parquet(eval_path)
    
    # 載入做多模型
    clf_long = xgb.XGBClassifier()
    clf_long.load_model(model_long_path)
    
    # 載入做空模型
    clf_short = xgb.XGBClassifier()
    clf_short.load_model(model_short_path)
except Exception as e:
    print(f"檔案讀取失敗: {e}")
    sys.exit()

feature_cols = clf_long.get_booster().feature_names
short_feature_cols = clf_short.get_booster().feature_names
if feature_cols and short_feature_cols and feature_cols != short_feature_cols:
    raise ValueError("Long and short XGBoost models use different feature columns.")
if not feature_cols:
    prob_cols = [c for c in df_eval.columns if c.startswith('prob_S')]
    feature_cols = ['z_t', 'c_t', 'a_t', 's_t', 'm_t', 'bias_60d', 'net_buy_amt_60d'] + prob_cols

# XGBoost 預測雙向機率
print("   -> 正在預測做多與做空機率...")
X_eval = df_eval[feature_cols]
df_eval['pred_prob_long'] = clf_long.predict_proba(X_eval)[:, 1]
df_eval['pred_prob_short'] = clf_short.predict_proba(X_eval)[:, 1]
df_eval['date'] = df_eval['date'].astype(str).str[:10]

# 計算做多機率的 EMA 平滑版本 (依個股分組，按日期排序)
print("   -> 正在計算做多機率 EMA 平滑 (span=3, 5, 10)...")
df_eval = df_eval.sort_values(['stock_id', 'date'])
for span in [3, 5, 10]:
    df_eval[f'pred_prob_long_ema{span}'] = (
        df_eval.groupby('stock_id')['pred_prob_long']
               .transform(lambda x, s=span: x.ewm(span=s, adjust=False, min_periods=1).mean())
    )

# 全域交易參數
INITIAL_CAPITAL = 1000000 
FEE_BUY = 0.001425        
FEE_SELL = 0.001425       
TAX_SELL = 0.003          

# ==========================================
# 2. 建立 K 線資料庫與 0050 Benchmark
# ==========================================
print("\n2. 預載 K 線資料庫與 0050 大盤...")
all_signal_stocks = set(df_eval['stock_id'].unique())
kline_cache = {}

for stock_id in tqdm(all_signal_stocks, desc="載入個股 K 線"):
    kdf = load_stock_data(stock_info_dir, stock_id)
    if not kdf.empty:
        kdf['date'] = kdf['date'].astype(str).str[:10]
        kdf = kdf.sort_values('date').set_index('date')
        kline_cache[stock_id] = kdf

df_0050 = None
df_0050 = load_stock_data(stock_info_dir, "0050")
if not df_0050.empty:
    df_0050['date'] = df_0050['date'].astype(str).str[:10]
    df_0050 = df_0050.sort_values('date').set_index('date')
    
    # 0050 股價還原處理 (依據 6/18 切割基準)
    SPLIT_DATE = '2025-06-18' 
    if SPLIT_DATE in df_0050.index or df_0050.index.max() >= SPLIT_DATE:
        df_0050.loc[df_0050.index >= SPLIT_DATE, 'close'] *= 4
    print("[OK] 成功載入 0050 作為大盤基準線。")

# ==========================================
# 3. 共用回測引擎函數 (新增買賣價格與模型機率紀錄)
# ==========================================
all_dates = sorted(df_eval['date'].unique())

def run_top_n_backtest(eval_df, n, long_threshold=0.6, short_threshold=0.6, trailing_stop_ratio=0.8, short_prob_col='pred_prob_short', long_prob_col='pred_prob_long'):
    cash = INITIAL_CAPITAL
    current_holdings = {}  
    equity_curve = []   
    trade_history = []  

    def get_next_open_price(stock_id, current_date):
        if stock_id in kline_cache:
            stock_dates = kline_cache[stock_id].index
            future_dates = stock_dates[stock_dates > current_date]
            if not future_dates.empty:
                next_date = future_dates[0] 
                return kline_cache[stock_id].loc[next_date]['open'], next_date
        return None, None

    for today in all_dates:
        # ------------------------------------------
        # A. 盤中防護：固定停損 (高點回落 20%)
        # ------------------------------------------
        stocks_to_remove = []
        for stock_id, holding in current_holdings.items():
            if stock_id in kline_cache and today in kline_cache[stock_id].index:
                row = kline_cache[stock_id].loc[today]
                
                if row['max'] > holding['highest_price']:
                    holding['highest_price'] = row['max']
                    
                stop_price = holding['highest_price'] * trailing_stop_ratio
                
                if row['min'] <= stop_price:
                    sell_price = row['open'] if row['open'] < stop_price else stop_price
                    
                    gross_proceeds = holding['shares'] * sell_price
                    net_proceeds = gross_proceeds * (1 - FEE_SELL - TAX_SELL) 
                    cash += net_proceeds
                    buy_cost = holding['shares'] * holding['buy_price'] * (1 + FEE_BUY)
                    
                    trade_history.append({
                        '股票代號': stock_id, 
                        '買進日期': holding['buy_date'], 
                        '賣出日期': today, 
                        '買進價格': round(holding['buy_price'], 2),
                        '賣出價格': round(sell_price, 2),
                        '做多機率(買進時)': round(holding['buy_prob_long'], 4),
                        '做空機率(賣出時)': 'N/A (價格觸發)', # 盤中價格觸發，非模型觸發
                        '報酬率': round((net_proceeds / buy_cost) - 1, 4), 
                        '賣出原因': '高點回落20%停損'
                    })
                    stocks_to_remove.append(stock_id)
        
        for stock_id in stocks_to_remove:
            del current_holdings[stock_id]

        # ------------------------------------------
        # B. 盤後決策：雙模型判斷 (隔日開盤執行)
        # ------------------------------------------
        today_preds = eval_df[eval_df['date'] == today]
        
        if not today_preds.empty:
            
            # [B-1] 檢查現有持股是否觸發「做空模型」出場
            stocks_to_sell_by_model = []
            for stock_id, holding in current_holdings.items():
                stock_pred = today_preds[today_preds['stock_id'] == stock_id]
                if not stock_pred.empty:
                    short_prob = stock_pred.iloc[0][short_prob_col]
                    if short_prob >= short_threshold:
                        stocks_to_sell_by_model.append((stock_id, short_prob)) # 紀錄觸發的做空機率
            
            for stock_id, short_prob in stocks_to_sell_by_model:
                sell_price, actual_sell_date = get_next_open_price(stock_id, today)
                if sell_price is not None:
                    gross_proceeds = current_holdings[stock_id]['shares'] * sell_price
                    net_proceeds = gross_proceeds * (1 - FEE_SELL - TAX_SELL)
                    cash += net_proceeds
                    buy_cost = current_holdings[stock_id]['shares'] * current_holdings[stock_id]['buy_price'] * (1 + FEE_BUY)
                    
                    trade_history.append({
                        '股票代號': stock_id, 
                        '買進日期': current_holdings[stock_id]['buy_date'], 
                        '賣出日期': actual_sell_date,
                        '買進價格': round(current_holdings[stock_id]['buy_price'], 2),
                        '賣出價格': round(sell_price, 2),
                        '做多機率(買進時)': round(current_holdings[stock_id]['buy_prob_long'], 4),
                        '做空機率(賣出時)': round(short_prob, 4),
                        '報酬率': round((net_proceeds / buy_cost) - 1, 4), 
                        '賣出原因': '做空模型預警賣出'
                    })
                    del current_holdings[stock_id]

            # [B-2] 做多模型檢查：買入與汰弱留強
            long_candidates = today_preds[today_preds[long_prob_col] >= long_threshold + 0.02].copy()

            for stock_id, holding in current_holdings.items():
                if stock_id in long_candidates['stock_id'].values:
                    new_prob = long_candidates[long_candidates['stock_id'] == stock_id].iloc[0][long_prob_col]
                    if new_prob > holding['buy_prob_long']:
                        holding['buy_prob_long'] = new_prob

            candidates = long_candidates[~long_candidates['stock_id'].isin(current_holdings.keys())]
            candidates = candidates.sort_values(long_prob_col, ascending=False)

            for _, candidate in candidates.iterrows():
                cand_id = candidate['stock_id']
                cand_prob = candidate[long_prob_col]
                
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
                                'buy_date': actual_buy_date, 
                                'buy_prob_long': cand_prob # 紀錄買進時的做多機率
                            }
                            
                else:
                    weakest_stock_id = min(current_holdings, key=lambda k: current_holdings[k]['buy_prob_long'])
                    weakest_holding = current_holdings[weakest_stock_id]
                    
                    if cand_prob > weakest_holding['buy_prob_long']:
                        sell_price, actual_sell_date = get_next_open_price(weakest_stock_id, today)
                        if sell_price is not None:
                            gross_proceeds = weakest_holding['shares'] * sell_price
                            net_proceeds = gross_proceeds * (1 - FEE_SELL - TAX_SELL)
                            cash += net_proceeds
                            buy_cost = weakest_holding['shares'] * weakest_holding['buy_price'] * (1 + FEE_BUY)
                            
                            trade_history.append({
                                '股票代號': weakest_stock_id, 
                                '買進日期': weakest_holding['buy_date'], 
                                '賣出日期': actual_sell_date,
                                '買進價格': round(weakest_holding['buy_price'], 2),
                                '賣出價格': round(sell_price, 2),
                                '做多機率(買進時)': round(weakest_holding['buy_prob_long'], 4),
                                '做空機率(賣出時)': 'N/A (被擠出)', 
                                '報酬率': round((net_proceeds / buy_cost) - 1, 4), 
                                '賣出原因': '滿檔換股'
                            })
                            del current_holdings[weakest_stock_id]
                            
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
                                        'buy_date': actual_buy_date, 
                                        'buy_prob_long': cand_prob
                                    }
                    else:
                        break 

        # ------------------------------------------
        # C. 每日結算帳戶總淨值
        # ------------------------------------------
        daily_equity = cash
        for stock_id, holding in current_holdings.items():
            if stock_id in kline_cache and today in kline_cache[stock_id].index:
                daily_equity += holding['shares'] * kline_cache[stock_id].loc[today]['close']
            else:
                daily_equity += holding['shares'] * holding['buy_price']
                
        equity_curve.append({'date': today, 'equity': daily_equity})

    # 期末強制平倉
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
            '股票代號': stock_id, 
            '買進日期': holding['buy_date'], 
            '賣出日期': last_date,
            '買進價格': round(holding['buy_price'], 2),
            '賣出價格': round(sell_price, 2),
            '做多機率(買進時)': round(holding['buy_prob_long'], 4),
            '做空機率(賣出時)': 'N/A',
            '報酬率': round((net_proceeds / buy_cost) - 1, 4), 
            '賣出原因': '期末平倉'
        })
        
    equity_df = pd.DataFrame(equity_curve)
    equity_df['date'] = pd.to_datetime(equity_df['date'])
    equity_df['cummax'] = equity_df['equity'].cummax()
    equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax']
    
    mdd = equity_df['drawdown'].min()
    total_return = (equity_df.iloc[-1]['equity'] / INITIAL_CAPITAL) - 1
    
    return equity_df, trade_history, total_return, mdd

# ==========================================
# 4. 執行多組 n 的回測與繪圖
# ==========================================
print("\n3. 執行多檔持股 (Top-N) 回測對決...")
n_values = [1, 3, 5]
results = {}

# 策略參數
LONG_PROB_THRESHOLD = 0.6
SHORT_PROB_THRESHOLD = 0.8
TRAILING_STOP = 0.8        

for n in n_values:
    print(f"-> 正在回測 Top-{n} 策略...")
    eq, trades, ret, mdd = run_top_n_backtest(
        eval_df=df_eval, 
        n=n, 
        long_threshold=LONG_PROB_THRESHOLD, 
        short_threshold=SHORT_PROB_THRESHOLD, 
        trailing_stop_ratio=TRAILING_STOP
    )
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

print("\n4. 繪製並儲存比較圖表...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 11), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
colors = ['#d62728', '#ff7f0e', '#2ca02c', '#9467bd', '#e377c2'] 

# 繪製淨值曲線
for i, n in enumerate(n_values):
    res = results[n]
    ax1.plot(res['equity']['date'], res['equity']['equity'], color=colors[i], linewidth=2.0,
             label=f'XGB Top-{n} (Total Return: {res["ret"]*100:.2f}%)')

if df_0050 is not None:
    ax1.plot(df_0050_eq['date'], df_0050_eq['equity'], color='#1f77b4', linewidth=1.5, linestyle='--',
             alpha=0.8, label=f'0050 Benchmark (Total Return: {benchmark_return*100:.2f}%)')

ax1.set_title('XGBoost Dual-Model (Top-N) Strategy Comparison vs 0050', fontsize=16, fontweight='bold')
ax1.set_ylabel('Portfolio Equity (TWD)', fontsize=12)
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
    
ax2.set_ylabel('Drawdown (%)', fontsize=12)
ax2.set_xlabel('Date', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(loc='lower left', fontsize=10, ncol=2)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)

plt.tight_layout()
save_path = output_dir / 'equity_curve_top_n_comparison.png'
plt.savefig(save_path, dpi=300)

# ==========================================
# 5. 輸出表格報告
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
row_equity += f" {int(df_0050_eq['equity'].iloc[-1]):>11,}" if df_0050 is not None else f" {'N/A':<11}"
print(row_equity)

# 總報酬率
row_ret = f"{'總報酬率':<12} |"
for n in n_values:
    row_ret += f" {results[n]['ret']*100:>9.2f}% |"
row_ret += f" {benchmark_return*100:>9.2f}%"
print(row_ret)

# 最大回撤
row_mdd = f"{'最大回撤(MDD)':<11} |"
for n in n_values:
    row_mdd += f" {results[n]['mdd']*100:>9.2f}% |"
row_mdd += f" {benchmark_mdd*100:>9.2f}%"
print(row_mdd)

# 交易次數
row_trades = f"{'總交易次數':<12} |"
for n in n_values:
    row_trades += f" {len(results[n]['trades']):>9} 次 |"
row_trades += f" {'N/A':>10}"
print(row_trades)

# 勝率
row_win_rate = f"{'實戰勝率':<12} |"
for n in n_values:
    trades = results[n]['trades']
    win_r = sum(1 for t in trades if t['報酬率'] > 0) / len(trades) if len(trades)>0 else 0
    row_win_rate += f" {win_r*100:>9.2f}% |"
row_win_rate += f" {'N/A':>10}"
print(row_win_rate)

print("="*85)
print(f"[OK] 圖表已成功儲存至: {save_path}")

# ==========================================
# 6. 輸出 Top-1 交易明細至 CSV
# ==========================================
print("\n6. 正在匯出 Top-1 策略的交易明細...")

# 從 results 字典中提取 n=1 的交易紀錄
top1_trades = results[1]['trades']

if len(top1_trades) > 0:
    df_top1_trades = pd.DataFrame(top1_trades)
    
    # 確保輸出目錄存在
    reports_dir = output_dir / 'reports'
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / 'top1_trade_history.csv'
    
    # 使用 utf-8-sig 編碼，避免在 Windows Excel 打開時中文變亂碼
    df_top1_trades.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"[OK] Top-1 交易明細已成功儲存至: {csv_path}")
else:
    print("[WARN] Top-1 策略在此期間內沒有任何交易紀錄。")

# ==========================================
# 7. EMA 平滑做多機率 對比回測 (Top-1, Top-3, Top-5)
# ==========================================
print("\n7. 執行 EMA 做多機率平滑策略對比 (Top-1, Top-3, Top-5)...")

ema_variants = {
    'No Smoothing': 'pred_prob_long',
    'EMA-3':        'pred_prob_long_ema3',
    'EMA-5':        'pred_prob_long_ema5',
    'EMA-10':       'pred_prob_long_ema10',
}
ema_colors = ['#d62728', '#ff7f0e', '#2ca02c', '#9467bd']
EMA_N_VALUES = [1, 3, 5]

# 執行所有 (Top-N × EMA variant) 組合的回測
ema_all_results = {}  # key: (n, label)
for ema_n in EMA_N_VALUES:
    for label, col in ema_variants.items():
        print(f"   -> Top-{ema_n} × {label} 正在回測...")
        eq, trades, ret, mdd = run_top_n_backtest(
            eval_df=df_eval,
            n=ema_n,
            long_threshold=LONG_PROB_THRESHOLD,
            short_threshold=SHORT_PROB_THRESHOLD,
            trailing_stop_ratio=TRAILING_STOP,
            long_prob_col=col,
        )
        ema_all_results[(ema_n, label)] = {'equity': eq, 'trades': trades, 'ret': ret, 'mdd': mdd}

# 8. 繪製並儲存 EMA 對比圖表 (每個 Top-N 一張)
print("\n8. 繪製並儲存 EMA 對比圖表...")
for ema_n in EMA_N_VALUES:
    fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(16, 11), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

    for i, label in enumerate(ema_variants):
        res = ema_all_results[(ema_n, label)]
        ax3.plot(res['equity']['date'], res['equity']['equity'],
                 color=ema_colors[i], linewidth=2.0,
                 label=f'{label} (Total Return: {res["ret"]*100:.2f}%)')

    if df_0050 is not None:
        ax3.plot(df_0050_eq['date'], df_0050_eq['equity'],
                 color='#1f77b4', linewidth=1.5, linestyle='--', alpha=0.8,
                 label=f'0050 Benchmark (Total Return: {benchmark_return*100:.2f}%)')

    ax3.set_title(f'Long Prob EMA Smoothing Comparison (Top-{ema_n}) vs 0050', fontsize=16, fontweight='bold')
    ax3.set_ylabel('Portfolio Equity (TWD)', fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.legend(loc='upper left', fontsize=11)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

    for i, label in enumerate(ema_variants):
        res = ema_all_results[(ema_n, label)]
        ax4.plot(res['equity']['date'], res['equity']['drawdown'] * 100,
                 color=ema_colors[i], linewidth=1.2, alpha=0.8,
                 label=f'{label} MDD: {res["mdd"]*100:.1f}%')

    if df_0050 is not None:
        ax4.plot(df_0050_eq['date'], df_0050_eq['drawdown'] * 100,
                 color='#1f77b4', linewidth=1.5, linestyle='--', alpha=0.8,
                 label=f'0050 MDD: {benchmark_mdd*100:.1f}%')

    ax4.set_ylabel('Drawdown (%)', fontsize=12)
    ax4.set_xlabel('Date', fontsize=12)
    ax4.grid(True, linestyle='--', alpha=0.6)
    ax4.legend(loc='lower left', fontsize=10, ncol=2)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)

    plt.tight_layout()
    ema_save_path = output_dir / f'equity_curve_ema_long_top{ema_n}_comparison.png'
    plt.savefig(ema_save_path, dpi=300)
    print(f"   [OK] Top-{ema_n} 圖表已儲存至: {ema_save_path}")
    plt.close(fig2)

# 輸出 EMA 對比表格 (每個 Top-N 一個區塊)
for ema_n in EMA_N_VALUES:
    print("\n" + "="*90)
    print(f" {'EMA 做多平滑策略對比結果 (Top-'+str(ema_n)+', 初始 100 萬)':^85} ")
    print("="*90)

    header2 = f"{'指標':<14} |"
    for label in ema_variants:
        header2 += f" {label:<14} |"
    header2 += f" {'0050大盤':<11}"
    print(header2)
    print("-" * 90)

    row_eq2 = f"{'期末淨值':<12} |"
    for label in ema_variants:
        res = ema_all_results[(ema_n, label)]
        row_eq2 += f" {int(res['equity'].iloc[-1]['equity']):<14,} |"
    row_eq2 += f" {int(df_0050_eq['equity'].iloc[-1]):>11,}" if df_0050 is not None else f" {'N/A':<11}"
    print(row_eq2)

    row_ret2 = f"{'總報酬率':<12} |"
    for label in ema_variants:
        res = ema_all_results[(ema_n, label)]
        row_ret2 += f" {res['ret']*100:>12.2f}% |"
    row_ret2 += f" {benchmark_return*100:>9.2f}%"
    print(row_ret2)

    row_mdd2 = f"{'最大回撤(MDD)':<11} |"
    for label in ema_variants:
        res = ema_all_results[(ema_n, label)]
        row_mdd2 += f" {res['mdd']*100:>12.2f}% |"
    row_mdd2 += f" {benchmark_mdd*100:>9.2f}%"
    print(row_mdd2)

    row_trades2 = f"{'總交易次數':<12} |"
    for label in ema_variants:
        res = ema_all_results[(ema_n, label)]
        row_trades2 += f" {len(res['trades']):>11} 次 |"
    row_trades2 += f" {'N/A':>10}"
    print(row_trades2)

    row_wr2 = f"{'實戰勝率':<12} |"
    for label in ema_variants:
        res = ema_all_results[(ema_n, label)]
        trades = res['trades']
        wr = sum(1 for t in trades if t['報酬率'] > 0) / len(trades) if trades else 0
        row_wr2 += f" {wr*100:>12.2f}% |"
    row_wr2 += f" {'N/A':>10}"
    print(row_wr2)

    print("="*90)

# 顯示圖表
plt.show()
