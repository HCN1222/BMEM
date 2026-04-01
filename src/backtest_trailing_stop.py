import pandas as pd
import numpy as np
import xgboost as xgb
import os
import sys

# 開啟 Pandas 對齊支援
pd.set_option('display.unicode.east_asian_width', True)

print("1. 正在載入 XGBoost 模型與 Eval 測試資料...")
# ==========================================
# 1. 載入模型與資料
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

# 定義特徵 (與訓練時一致)
prob_cols = [f'prob_S{i}' for i in range(10)]
feature_cols = ['z_t', 'c_t', 'a_t', 's_t', 'm_t', 'bias_60d', 'net_buy_amt_60d'] + prob_cols

# 預測大漲機率
X_eval = df_eval[feature_cols]
df_eval['pred_prob'] = clf.predict_proba(X_eval)[:, 1]

# 篩選大於門檻的進場訊號
THRESHOLD = 0.70
signals_df = df_eval[df_eval['pred_prob'] >= THRESHOLD].copy()
signals_df['date'] = signals_df['date'].astype(str).str[:10]

unique_stocks = signals_df['stock_id'].unique()
print(f"-> 在 Eval 資料中，門檻 >= {THRESHOLD} 共有 {len(signals_df)} 個買進訊號，分布於 {len(unique_stocks)} 檔股票。")
print("\n2. 開始逐日進行移動停損回測 (Trailing Stop-Loss 10%)...")

# ==========================================
# 2. 移動停損回測引擎 (Event-Driven Backtest)
# ==========================================
trade_history = []

for stock_id in unique_stocks:
    # 取得這檔股票所有的買進訊號日期
    stock_signals = signals_df[signals_df['stock_id'] == stock_id]['date'].tolist()
    
    # 載入這檔股票完整的 K 線資料來模擬走勢
    kline_path = f"./data/stocks/{stock_id}_2021-06-30_to_2026-02-11.parquet"
    if not os.path.exists(kline_path):
        continue
        
    kdf = pd.read_parquet(kline_path)
    kdf['date'] = kdf['date'].astype(str).str[:10]
    # 確保依日期排序
    kdf = kdf.sort_values('date').reset_index(drop=True)
    
    # 交易狀態變數
    holding = False
    buy_date = None
    buy_price = 0.0
    highest_price = 0.0
    
    for idx, row in kdf.iterrows():
        current_date = row['date']
        current_open = row['open']
        current_close = row['close']
        current_high = row['max']
        current_low = row['min']
        
        if holding:
            # 1. 更新持有期間的最高價 (High-Water Mark)
            if current_high > highest_price:
                highest_price = current_high
                
            # 2. 計算目前的移動停損價位
            stop_price = highest_price * 0.8
            
            # 3. 檢查是否觸發停損 (盤中最低價跌破停損價)
            if current_low <= stop_price:
                # 實戰防呆：如果開盤就直接跳空跌破停損，只能賣在開盤價；否則賣在停損價
                sell_price = current_open if current_open < stop_price else stop_price
                return_pct = (sell_price / buy_price) - 1
                
                trade_history.append({
                    '股票代號': stock_id,
                    '買進日期': buy_date,
                    '賣出日期': current_date,
                    '買進價': round(buy_price, 2),
                    '最高價': round(highest_price, 2),
                    '賣出價': round(sell_price, 2),
                    '報酬率': return_pct,
                    '備註': '移動停損出場'
                })
                holding = False # 恢復空手狀態
                
        else: # 如果目前空手，檢查今天有沒有買進訊號
            if current_date in stock_signals:
                holding = True
                buy_date = current_date
                # 假設在發出訊號當天的收盤價買進
                buy_price = current_close 
                highest_price = current_close

    # 如果到資料集最後一天還持有，強制平倉結算
    if holding:
        last_row = kdf.iloc[-1]
        sell_price = last_row['close']
        return_pct = (sell_price / buy_price) - 1
        trade_history.append({
            '股票代號': stock_id,
            '買進日期': buy_date,
            '賣出日期': last_row['date'],
            '買進價': round(buy_price, 2),
            '最高價': round(highest_price, 2),
            '賣出價': round(sell_price, 2),
            '報酬率': return_pct,
            '備註': '期末強制平倉'
        })
print("3. 產出交易明細與績效總表...\n")
# ==========================================
# 3. 輸出交易結果與績效統計
# ==========================================
trades_df = pd.DataFrame(trade_history)

if len(trades_df) == 0:
    print("沒有任何交易發生！可能是門檻設太高或 Eval 資料時間過短。")
    sys.exit()

# 為了計算 All-in 複利，我們先將交易按時間排序
trades_df = trades_df.sort_values('買進日期').reset_index(drop=True)

# 計算基本統計指標
total_trades = len(trades_df)
winning_trades = len(trades_df[trades_df['報酬率'] > 0])
win_rate = winning_trades / total_trades if total_trades > 0 else 0
avg_return = trades_df['報酬率'].mean()
max_win = trades_df['報酬率'].max()
max_loss = trades_df['報酬率'].min()

# 計算 All-in 總累積報酬率 (連乘效應)
# 公式：(1 + R1) * (1 + R2) * ... * (1 + Rn) - 1
total_compounded_return = np.prod(1 + trades_df['報酬率']) - 1

# 格式化輸出用的表格
display_df = trades_df.copy()
display_df['報酬率'] = (display_df['報酬率'] * 100).round(2).astype(str) + '%'

print("="*95)
print("                               移動停損實戰回測明細 (Eval Set)")
print("="*95)
# 因為明細可能很長，如果超過 50 筆，我們只印出前 25 筆跟最後 25 筆
if len(display_df) > 50:
    print(display_df.head(25).to_markdown(index=False))
    print("\n... (中間省略) ...\n")
    print(display_df.tail(25).to_markdown(index=False))
else:
    print(display_df.to_markdown(index=False))

print("\n" + "="*50)
print("             回測績效總結")
print("="*50)
print(f"總交易次數 : {total_trades} 次")
print(f"獲利交易數 : {winning_trades} 次")
print(f"實戰勝率   : {win_rate*100:.2f}%")
print(f"平均報酬率 : {avg_return*100:.2f}%")
print(f"單筆最大賺 : {max_win*100:.2f}%")
print(f"單筆最大賠 : {max_loss*100:.2f}%")
print("-" * 50)
print(f"🚀 All-In 總累積報酬率 : {total_compounded_return*100:.2f}%")
print("="*50)

# 將完整明細存檔
os.makedirs('./outputs/backtest', exist_ok=True)
trades_df.to_csv('./outputs/backtest/trailing_stop_trades.csv', index=False, encoding='utf-8-sig')
print("\n✅ 完整交易明細已儲存至: ./outputs/backtest/trailing_stop_trades.csv")