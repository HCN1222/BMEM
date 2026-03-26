# BMEM

**Believe Merrill Enrich the Meal**
Step1. 利用HMM分析外資(美林)`建倉/持倉/清倉` 狀態，縮小選股範圍
Step2. 如果順利再利用Gradient Boosted Decision Tree分析多個外資狀態與股價關係

## Data
資料使用Finmind API下載
[Finmind](https://finmindtrade.com/analysis/#/data/api)
籌碼面資料缺失

![籌碼面資料缺失](./img/image.png)

### 目前已下載的資料集

時間皆為 2021/02/30 ~ 2026/02/11

1.  `./data/brokers/卷商分點編號`: 此資料夾中的`.parquet`包含十個外資的分點進出資料(每檔股票淨買賣)
2.  `./data/stocks`: 這裡包含上述分點交易所觸及的所有股票，檔案開頭為股票代碼


## Setup
1. add your **API KEY** in `.env.example` and rename the file as `.env`

### Tools for Downloading data
指令可參考`./script`下面的`.ps1`
- `download_broker_activity.py`: 抓取卷商分點資料
- `download_stock_info.py`: 抓取股票資料

## HMM

### States

There are three states: 
- **Building Position (建倉)**
- **Liquidating Position (清倉)**
- **Holding Position (持倉)**

### Observation Vector

採用 Gaussian Emission

$x_t = [z_t, a_t, s_t, I_t]$

符號解釋:
- $buy_t$ / $sell_t$: 第t天買總量/賣總量
- $V_t$: 第t天股票交易總量
- $C_t$: 第t天收盤價


---
#### 1. Normalized Net Buy (Flow Strength) $z_t$

$$z_t = \frac{nb_t}{V_t}= \frac{buy_t - sell_t}{V_t}$$
- 橫截面正規化: 為了要可以多股票一起訓練

#### 2. Logarithmic return $r_t$

$$r_t = \frac{C_t}{C_{t-1}}$$
- 觀察到有時卷商單純“漲就買、跌就賣”

#### 3. Activity level $a_t$

$$a_t = \frac{(|buy_t|+|sell_t|)}{V_t}$$
- 提供對沖資訊

#### 4. Average Normalized Net Buy (5 days) $m_5$

$$m_5 = \frac{1}{5} \sum_{t=T-4}^{T} z_t$$
- 希望利用observation vector 來補足短期記憶(first-order markov model 只根據前一個狀態來轉移)

#### 5. Directional Persistence (5 days) $s_t$

$$s_t = \frac{1}{5} \sum_{i=t-4}^{t} \text{sign}(nb_{i})$$
- Captures sustained buying or selling behavior.

#### 6. Flow Price Alignment (5 days) $c_t$

---

### Ablation study

1. 對時序面做正規化(z-score)容易造成模型誤判，以20日 rolling window 做正規化為例:第一天大量買超但剩餘19天毫無動作，此舉會造成正規化後的剩餘19天皆處於賣超狀態

**Not yet tested**
Flow-Price alignment 與 covariance 交互

1. 不加交互項 + full covariance
2. 加$sign(𝑟)⋅𝑧$ + diag covariance
3. 加$sign(𝑟)⋅sign(𝑧)$ + diag covariance

### Result

#### 測試結果

 - 存放位置: `./outputs/result_20260326_104333/states_5`
 - 以下是GPT解說

本模型使用 5 維特徵矩陣 `[z_t, r_t, a_t, s_t, m_t]` 訓練 Gaussian HMM，經過 BIC (Bayesian Information Criterion) 評估後，最佳狀態數為 5。模型成功從券商交易行為中，完美分離出具有高度物理意義的市場微結構，包含極端精準的「連續買超天數規律」。

![BIC_evaluation_curve](./outputs/result_20260326_104333/bic_evaluation_curve.png)

| State | Market Behavior (市場行為) | Key Feature Signatures (關鍵特徵表現) | Physical Meaning & Analysis (物理意義與分析) |
| :---: | :--- | :--- | :--- |
| **3** | 💤 **Inactive / Rest**<br>(觀望與休眠期) | 所有特徵平均值與標準差皆為 `0.000` | **券商空手，無交易動作。**<br>模型完美的底層過濾器，自動將無交易紀錄（補 0）的休市或觀望日歸類於此，避免干擾其他交易狀態的變異數。 |
| **2** | 🔥 **Aggressive Accumulation**<br>(極端強勢建倉) | **$s_t = 1.000$ (Std = 0.000)**<br>$z_t = 0.052$ (最高單日買超)<br>$m_t = 0.053$ (最高5日均買) | **機構法人連續 5 天無腦做多。**<br>$s_t=1$ 在數學上代表過去 5 天「每一天都在淨買超」。這顯示了極度強勢的波段吃貨行為，且通常伴隨著最高的大盤上漲報酬 ($r_t = 0.003$)。 |
| **4** | 📈 **Steady Accumulation**<br>(穩健建倉/逢低承接) | **$s_t = 0.600$ (Std = 0.000)**<br>$a_t = 0.070$ (高活躍度)<br>$z_t = 0.027$ (溫和買超) | **有紀律的波段買盤（進 4 退 1）。**<br>$s_t=0.6$ 代表過去 5 天中有「4 天買超、1 天賣超」。相較於 State 2 的暴力拉升，這代表更聰明的拉回買進 (Buy on dips) 演算法或有耐心的建倉策略。 |
| **0** | 🌪️ **Aggressive Distribution**<br>(高壓出貨/劇烈震盪) | **$a_t = 0.101$ (全場最高活躍度)**<br>$s_t = -0.221$ (高變異數 0.429)<br>$z_t = -0.013$ (淨賣超) | **伴隨巨大成交量的倒貨或多空交戰。**<br>活躍度極高但方向性震盪，代表券商同時有大量買賣單對敲（當沖或洗盤換手），但最終結算為實質淨賣出，為市場危險訊號（出量下跌）。 |
| **1** | 📉 **Weak Distribution**<br>(弱勢調節/散戶雜訊) | $a_t = 0.033$ (低活躍度)<br>$z_t = -0.003$ (微弱賣超)<br>$s_t = -0.141$ (方向不明確) | **缺乏流動性的溫水煮青蛙或散戶賣壓。**<br>典型的「垃圾時間 (Garbage Time)」。券商偶爾零星賣出，交易不熱絡，對大盤價格幾乎沒有影響力 ($r_t = 0.000$)。 |

> **💡 Note on Features:** > * $z_t$: Normalized Net Buy (單日買賣超比例)
> * $r_t$: Logarithmic Return (對數報酬率)
> * $a_t$: Activity Level (市場活躍度/週轉率)
> * $s_t$: Directional Persistence (近5日方向持續性)
> * $m_t$: 5-day Moving Average of $z_t$ (近5日買賣超動能)



# 下面是筆記
