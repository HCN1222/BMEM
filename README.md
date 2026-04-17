# BMEM

**Believe Merrill Enrich the Meal**
Step1. 利用HMM分析外資(美林)`建倉/持倉/清倉` 狀態，縮小選股範圍
Step2. 利用Gradient Boosted Decision Tree分析多個外資狀態與股價關係

## SRC 檔案供用(未整理)

打 x 表示現在已經沒什麼用，是當初研發時初步簡單的測試而已，已經被更複雜的測試取代
- `__init__.py`: package file
- `download_broker_activity`: 一開始下載broker資料的script
- `download_stock_info`: 一開始下載stock資料的script
- `evaluate_states`: 利用迴圈跑BIC方便找出HMM最適合的狀態數
- `evaluate_strategy_test`: rolling verterbi解碼 + 給定條件下測試每個state勝率(x)
- `portfolio_backtest.py`: 以總資金100萬下去測試XGBoost與EDA，並與0050做比較
- `prepare_xgb_data.py`: 利用現有HMM模型，將資訊以及狀態(機率)與target打包成表格供XGBoost用
- `preprocess.py`: 過濾有問題的資料(包含0, NaN等骯髒的資料)，並且將資料切分為train 跟eval，然後把特徵抽取出來打包給HMM訓練用
- `train_XGBoost.py`: 訓練XGBoost
- `train.py`: 訓練HMM
- `visualize_output.py`: 視覺化顯示HMM分類(一開始檢視HMM分的理不理想用)

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

#### 6. Flow Price Alignment $c_t$

### Result

##### EXP0
嘗試了各種feature set，發現以下問題:
1. 當對當日報酬以及淨買超做時序面正規化(z-score)容易造成模型誤判，以20日 rolling window 做正規化為例:第一天大量買超但剩餘19天毫無動作，此舉會造成正規化後的剩餘19天皆處於賣超狀態
2. 當參數過少(如只有m_t)會導致模型過於簡化，沒有發揮到多維度資料處理的長處，直接用買賣超硬條件即可
3. 當state數過少，會造成obsercation feature的標準差超大，分類失敗

##### EXP1
- 特徵矩陣: `[z_t, r_t, a_t, s_t, m_t]`
- 關閉時序面標準化 `--disable_standardize`
- 跑states = 2~6，發現state = 5 時 BIC最小

**特徵矩陣中的 `r_t`在五個state中都一樣，判斷為沒有用**

##### EXP2
- 與EXP1 配置相同，但是把**r_t**改成**c_t**(flow-price alignment)
- 特徵矩陣: `[z_t, c_t, a_t, s_t, m_t]`
- 算法是sign(r_t) * sign(z_t)，但這樣狀態數太少只有[-1,0,1]
- 跑states = 2~6，發現state = 6 時 BIC最小
但是在state=6時，log likelihood不收斂，不斷跳上跳下

另外發現在state數只有2跟3時，分別跌代4次跟2次及沒有improvement了

**觀察發現模型不會進一步細分`flow-price alignment`與漲跌/買進的進一步關係(ex:把追漲+殺跌混再一起)**

##### EXP3
- 把c_t改成五個狀態[-2,-1,0,1,2]，依序分別代表(跌+賣/漲+賣/沒操作/跌+買/漲+買)
- 並且發現在state = 10時，達到BIC的轉折點，並且收斂得很好，log likelihood也最大

**不過state數太多很難解釋，打算直接喂給XBoost，找尋state與漲幅關係**

# 下面是筆記
