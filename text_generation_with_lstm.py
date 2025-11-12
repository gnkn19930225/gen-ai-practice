# %%
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import TextVectorization

def reweight_distribution(original_distribution, temperature=0.5):
    """
    根據溫度參數重新調整概率分佈。

    參數:
        original_distribution: 原始概率分佈（numpy array）
        temperature: 溫度參數（預設 0.5）
                    - T < 1.0：低溫，分佈更尖銳，接近貪心採樣
                    - T = 1.0：標準溫度，結果與原本相同（不改變）
                    - T > 1.0：高溫，分佈更平緩，各元素選中機率更接近，更隨機

    返回:
        調整後的概率分佈（總和為 1）

    例子:
        original_distribution = [0.7, 0.2, 0.05, 0.05]
        - T = 0.5:  [0.88, 0.10, 0.01, 0.01]  # 更尖銳
        - T = 1.0:  [0.73, 0.20, 0.04, 0.04]  # 保持原本
        - T = 2.0:  [0.54, 0.28, 0.09, 0.09]  # 更平緩
    """
    # 步驟 1: 取自然對數並除以溫度
    # np.log() 是自然對數（ln），不是平方
    distribution = np.log(original_distribution) / temperature

    # 步驟 2: 計算指數
    # np.exp() 是自然指數函數 (e^x)，不是平方
    # 例如：exp(1) ≈ 2.71828，exp(2) ≈ 7.39
    distribution = np.exp(distribution)

    # 步驟 3: 正規化，使概率總和為 1
    return distribution / np.sum(distribution)


def prepare_lm_dataset(text_batch):
    vectorized_sequences = text_vectorization(text_batch)
    x = vectorized_sequences[:, :-1]
    y = vectorized_sequences[:, 1:]
    return x, y


sequence_length = 100
vocab_size = 15000


dataset = keras.utils.text_dataset_from_directory(
    directory="aclImdb", label_mode=None, batch_size=256)
dataset = dataset.map(lambda x: tf.strings.regex_replace(x, "<br />", " "))

text_vectorization = TextVectorization(
    max_tokens=vocab_size,  # 限制詞彙表大小，避免無止境擴張
    output_mode="int",  # 將文字轉成整數索引序列方便後續模型使用
    output_sequence_length=sequence_length,  # 固定序列長度以便批次處理
)

text_vectorization.adapt(dataset)
lm_dataset = dataset.map(prepare_lm_dataset, num_parallel_calls=4)
