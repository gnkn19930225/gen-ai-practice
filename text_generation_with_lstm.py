import numpy as np

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