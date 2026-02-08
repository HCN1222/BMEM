# BMEM
## Believe Merlin Enjoy Michelin
分析外資的持倉狀態，縮小選股範圍
目標是波段交易

## API
[Finmind](https://finmindtrade.com/analysis/#/data/api)

### 預計進度
1. 先跑 Baseline 0（事件日買超 → T+1 進 → 固定持有）
2. 股票池限制在流動性高的（避免滑價假象）
3. 跑 3 個持有期：5/10/20 天
4. 用超額報酬（扣大盤或產業）比較
5. 再加 Baseline 1（連續性）看是否更穩

### 設想
1. 利用HMM (sticky HMM)去分析外資目的(建倉/持倉)
2. 利用1.分析多加外資
接著再利用 Gradient Boosted Decision Tree