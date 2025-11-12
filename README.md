# 生成式深度學習練習

本資料夾用於生成式深度學習的練習和研究。

---

## 資料集來源

- `aclImdb_v1`：IMDB 影評情感分析資料集（來源：<https://ai.stanford.edu/~amaas/data/sentiment/>）

---

## 實作步驟

### 環境準備

1. **建立虛擬環境**（建議）
   ```bash
   python -m venv env
   ```

2. **啟動虛擬環境**
   - Windows：
     ```bash
     .\env\Scripts\Activate
     ```
   - macOS/Linux：
     ```bash
     source env/bin/activate
     ```

3. **安裝依賴套件**
   ```bash
   pip install -r requirements.txt
   ```

### 準備資料集

1. 下載 IMDB 資料集（來源：<https://ai.stanford.edu/~amaas/data/sentiment/>）
2. 解壓縮到專案目錄下，確保資料夾結構為 `aclImdb/`

### 執行文字生成模型

#### Transformer 文字生成

使用 Transformer 架構進行文字生成訓練：

```bash
python text_generation_with_transformer.py
```

---

## 訓練流程詳解

### 1. 資料載入與清理

```python
# 從 aclImdb 目錄載入文字資料
dataset = keras.utils.text_dataset_from_directory(
    directory="aclImdb",
    label_mode=None,      # 不需要標籤（生成任務）
    batch_size=256        # 每批次 256 個樣本
)

# 清理資料：移除 HTML 標籤
dataset = dataset.map(lambda x: tf.strings.regex_replace(x, "<br />", " "))
```

**說明：**
- 使用 IMDB 影評資料集，包含訓練和測試的正負評文字檔
- `label_mode=None` 因為這是無監督的文字生成任務，不需要情感標籤
- 使用正則表達式移除 HTML 換行標籤 `<br />`

### 2. 文字向量化（前處理）

```python
text_vectorization = TextVectorization(
    max_tokens=15000,                    # 限制詞彙表大小為 15,000 個最常見的詞
    output_mode="int",                   # 輸出整數索引序列
    output_sequence_length=100           # 固定序列長度為 100
)

# 分析整個資料集，建立詞彙表
text_vectorization.adapt(dataset)
```

**前處理步驟：**
1. **分詞（Tokenization）**：將文字分割成單詞
2. **建立詞彙表**：統計所有單詞，保留最常見的 15,000 個
3. **數值化**：將每個單詞轉換為唯一的整數索引
4. **填充/截斷**：統一序列長度為 100（不足補 0，超過則截斷）

**詞彙表結構：**
- 索引 0：保留給填充（padding）
- 索引 1：通常是 `[UNK]`（未知詞）
- 索引 2-15000：按詞頻排序的單詞

### 3. 訓練資料準備

```python
def prepare_lm_dataset(text_batch):
    vectorized_sequences = text_vectorization(text_batch)
    x = vectorized_sequences[:, :-1]  # 輸入：序列的前 99 個 token
    y = vectorized_sequences[:, 1:]   # 標籤：序列的後 99 個 token
    return x, y
```

**語言模型訓練原理：**

假設原始序列為：`["This", "movie", "is", "great"]`
向量化後：`[45, 123, 8, 567]`

```
輸入 (x):  [45, 123, 8]      → 預測 → 標籤 (y):  [123, 8, 567]
位置 0:     45               → 預測 →             123  (movie)
位置 1:     123              → 預測 →             8    (is)
位置 2:     8                → 預測 →             567  (great)
```

這種方式讓模型學習：**給定前面的詞，預測下一個詞**

**批次處理：**
```python
lm_dataset = dataset.map(prepare_lm_dataset, num_parallel_calls=4)
```
- 使用 4 個並行處理單元加速資料處理
- 每個批次包含 256 個訓練樣本

### 4. 模型架構

```python
# 定義輸入層，接受任意長度的整數序列（token 索引）
inputs = keras.Input(shape=(None,), dtype="int64")
# 位置編碼層：將 token 索引轉為 256 維向量並加上位置資訊
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(inputs)
# Transformer Decoder：透過多頭注意力機制學習序列的上下文關係
x = TransformerDecoder(embed_dim, latent_dim, num_heads)(x, x)
# 輸出層：將向量轉換為 15000 個單詞的概率分佈（使用 softmax）
outputs = layers.Dense(vocab_size, activation="softmax")(x)
# 建立模型：定義從輸入到輸出的完整計算圖
model = keras.Model(inputs, outputs)
```

**模型層級說明：**

1. **輸入層**：接受可變長度的整數序列
2. **位置編碼層（PositionalEmbedding）**：
   - Token Embedding：將整數索引轉換為 256 維向量
   - Position Embedding：為每個位置添加位置資訊
   - 兩者相加，讓模型理解詞序
3. **Transformer Decoder**：
   - Self-Attention：學習序列內部的依賴關係
   - Cross-Attention：整合上下文資訊
   - Feed-Forward Network：非線性轉換
   - 使用因果遮罩（Causal Mask）確保只能看到之前的詞
   - **為什麼傳入 (x, x)？** 因為這是 Decoder-Only 架構（類似 GPT），沒有 Encoder，所以兩個參數都傳入相同的序列，讓 Cross-Attention 實際上也變成 Self-Attention
4. **輸出層**：
   - Dense(15000) + Softmax：輸出 15,000 維的概率分佈
   - 每個維度代表一個詞的概率

**訓練參數：**
- 序列長度：100
- 詞彙表大小：15,000
- 嵌入維度：256
- 潛在維度：2,048
- 注意力頭數：2
- 訓練輪數：200
- 損失函數：Sparse Categorical Crossentropy
- 優化器：RMSprop

### 5. 測試與生成

#### 生成原理

```python
class TextGenerator(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # 每個 epoch 結束時測試生成能力
        for temperature in self.temperatures:
            sentence = self.prompt  # 起始提示詞，例如 "This movie"
            for i in range(self.generate_length):
                # 1. 將當前句子向量化
                tokenized_sentence = text_vectorization([sentence])
                # 2. 模型預測下一個詞的概率分佈
                predictions = self.model(tokenized_sentence)
                # 3. 使用溫度採樣選擇下一個詞
                next_token = sample_next(predictions[0, i, :], temperature)# 要有溫度^.^
                # 4. 轉換回文字並添加到句子
                sampled_token = tokens_index[next_token]
                sentence += " " + sampled_token
```

**自回歸生成流程：**

```
初始提示: "This movie"
↓
步驟 1: "This movie" → 預測 → "is"     → "This movie is"
步驟 2: "This movie is" → 預測 → "really" → "This movie is really"
步驟 3: "This movie is really" → 預測 → "good" → "This movie is really good"
...
```

#### 溫度採樣測試

```python
def sample_next(predictions, temperature=1.0):
    # 1. 對預測概率取對數並除以溫度
    predictions = np.log(predictions) / temperature
    # 2. 計算指數（重新計算概率）
    exp_preds = np.exp(predictions)
    # 3. 正規化（確保總和為 1）
    predictions = exp_preds / np.sum(exp_preds)
    # 4. 根據調整後的概率分佈隨機採樣
    probas = np.random.multinomial(1, predictions, 1)
    return np.argmax(probas)
```

**多溫度測試（0.2, 0.5, 0.7, 1.0, 1.5）：**

| 溫度 | 特性 | 生成效果 |
|------|------|---------|
| 0.2  | 極低溫，接近貪心採樣 | 最保守、最可預測、可能重複 |
| 0.5  | 低溫 | 較保守、句子連貫性高 |
| 0.7  | 中低溫 | 平衡創意與連貫性 |
| 1.0  | 標準溫度 | 模型原始分佈 |
| 1.5  | 高溫 | 更有創意、更隨機、可能不連貫 |

**測試時機：**
- 每個 epoch 結束時自動生成文字
- 使用固定提示詞 `"This movie"` 確保可比較性
- 生成 50 個 token（約 50 個單詞）

**輸出範例：**
```
Epoch 1/200
== Generating with temperature 0.2
This movie is a very good movie with a great story and great acting...

== Generating with temperature 1.5
This movie offers unexpected twists bizarre characters surreal atmosphere...
```

### 6. 評估標準

雖然程式中沒有明確的量化評估，但可以通過以下方式判斷訓練效果：

1. **訓練損失（Loss）**：
   - 持續下降表示模型在學習
   - 穩定後代表收斂

2. **生成文字品質**：
   - 語法正確性：是否符合英文語法
   - 語意連貫性：句子是否有意義
   - 多樣性：不同溫度下的變化

3. **比較不同溫度的輸出**：
   - 低溫應該產生更保守、更常見的句子
   - 高溫應該產生更多樣化的句子

---

## 採樣方法概念

在生成式模型中，採樣方法決定了模型如何選擇下一個令牌（token）或動作。以下介紹三個核心概念及其關係。

### 1. 貪心採樣 (Greedy Sampling)

**核心概念**：永遠選擇概率最高的選項。

- **確定性**：同樣的輸入永遠產生同樣的輸出
- **選擇方法**：每一步都挑選概率最高的令牌/動作
- **特點**：
  - 速度快，計算效率高
  - 容易產生重複、單調的輸出
  - 傾向選擇最"明顯"的續文
  - 沒有隨機性

**例子**：如果概率是 `[0.7, 0.2, 0.05, 0.05]`，永遠挑第一個選項。

---

### 2. 隨機採樣 (Stochastic Sampling)

**核心概念**：根據概率分佈隨機選擇選項。

- **非確定性**：相同的輸入可能產生不同的輸出
- **選擇方法**：根據概率分佈進行採樣（尊重各選項的概率比例）
- **特點**：
  - 輸出多樣化，更有創意
  - 可能產生意外但合理的續文
  - 速度比貪心採樣慢
  - 包含隨機性/變異性

**例子**：概率 `[0.7, 0.2, 0.05, 0.05]`，70% 機率挑第一個，20% 挑第二個，以此類推。

---

### 3. Softmax Temperature (溫度)

**溫度** 是一個控制概率分佈"銳度"的參數。

#### 溫度運作原理

原始邏輯分數通過 softmax 轉換為概率：

```
P(token_i) = exp(logits_i / T) / Σ exp(logits_j / T)
```

其中 **T** 就是溫度。

#### 三種溫度情況

**1. T = 1.0（標準溫度）**
- 原始的 softmax 分佈
- 保持模型的自然概率分佈

**2. T < 1.0（低溫）**
- 概率分佈變得**更尖銳、更集中**
- 高概率選項的差異被放大
- 效果接近**貪心採樣**
- 輸出更確定、更保守

範例：`[0.7, 0.2, 0.05, 0.05]` → `[0.92, 0.07, 0.005, 0.005]`

**3. T > 1.0（高溫）**
- 概率分佈變得**更平緩、更均勻**
- 各選項的概率差異被縮小
- 所有選項更有機會被選中
- 輸出更隨機、更多樣化

範例：`[0.7, 0.2, 0.05, 0.05]` → `[0.54, 0.30, 0.08, 0.08]`

---

## 三者的整體關係

| 面向 | 貪心採樣 | 隨機採樣 |
|------|---------|---------|
| **確定性** | 確定性 | 非確定性 |
| **多樣性** | 低（重複） | 高（多變） |
| **速度** | 快 | 慢 |
| **使用場景** | 翻譯、摘要 | 創意寫作、對話 |
| **變異度** | 零 | 高 |

### 熵與隨機性

- 熵越大代表概率分佈越均勻，模型的選擇越沒有定論，其輸出的隨機性也就越高。

### 溫度與採樣的結合

```
低溫 (T<1)  →  貪心採樣效果
   ↓
標準溫度 (T=1)  →  隨機採樣（自然分佈）
   ↓
高溫 (T>1)  →  隨機採樣（高度隨機）
```

---

## 實際應用舉例

假設 logits = `[2.0, 1.0, 0.0, 0.0]`

### T = 0.5（低溫 - 保守）
```
P = [0.88, 0.10, 0.01, 0.01]  ← 貪心傾向強
```

### T = 1.0（標準 - 平衡）
```
P = [0.73, 0.20, 0.04, 0.04]  ← 正常分佈
```

### T = 2.0（高溫 - 創意）
```
P = [0.54, 0.28, 0.09, 0.09]  ← 更平均，隨機性高
```

---

## 常見的隨機採樣方法

- **Random Sampling**：直接從分佈中採樣
- **Top-k Sampling**：只從概率最高的 k 個令牌中採樣
- **Nucleus (Top-p) Sampling**：從累積概率超過 p 的最小令牌集中採樣
- **Temperature Sampling**：調整溫度來控制隨機程度

---

## 總結

- **貪心採樣** = 低溫 + 確定性選擇
- **隨機採樣** = 溫度可調 + 概率分佈採樣
- **溫度** 是調整「確定性」vs「多樣性」的關鍵參數
