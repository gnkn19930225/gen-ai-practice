import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
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

def sample_next(predictions, temperature=1.0):
    predictions = np.asarray(predictions).astype("float64")
    predictions = np.log(predictions) / temperature
    exp_preds = np.exp(predictions)
    predictions = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, predictions, 1)
    return np.argmax(probas)

class TextGenerator(keras.callbacks.Callback):
    def __init__(self,
                 prompt,
                 generate_length,
                 model_input_length,
                 temperatures=(1.,),
                 print_freq=1):
        self.prompt = prompt
        self.generate_length = generate_length
        self.model_input_length = model_input_length
        self.temperatures = temperatures
        self.print_freq = print_freq

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.print_freq != 0:
            return
        for temperature in self.temperatures:
            print("== Generating with temperature", temperature)
            sentence = self.prompt
            for i in range(self.generate_length):
                tokenized_sentence = text_vectorization([sentence])
                predictions = self.model(tokenized_sentence)
                next_token = sample_next(predictions[0, i, :])
                sampled_token = tokens_index[next_token]
                sentence += " " + sampled_token
            print(sentence)

class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=input_dim, output_dim=output_dim)
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim)
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.output_dim = output_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        # 使用 TensorFlow 操作來計算遮罩
        return tf.not_equal(inputs, 0)

    def get_config(self):
        config = super(PositionalEmbedding, self).get_config()
        config.update({
            "output_dim": self.output_dim,
            "sequence_length": self.sequence_length,
            "input_dim": self.input_dim,
        })
        return config


class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
          num_heads=num_heads, key_dim=embed_dim)
        self.attention_2 = layers.MultiHeadAttention(
          num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"),
             layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.supports_masking = True

    def get_config(self):
        config = super(TransformerDecoder, self).get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim,
        })
        return config

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1),
             tf.constant([1, 1], dtype=tf.int32)], axis=0)
        return tf.tile(mask, mult)

    def call(self, inputs, encoder_outputs, mask=None):
        # 取得因果遮罩（causal mask），確保每個位置只能看到它之前的位置
        causal_mask = self.get_causal_attention_mask(inputs)
        # 若有提供 mask，則結合 padding mask 與 causal mask
        if mask is not None:
            padding_mask = tf.cast(
                mask[:, tf.newaxis, :], dtype="int32")
            padding_mask = tf.minimum(padding_mask, causal_mask)
        else:
            # 沒有 mask 時直接使用 causal_mask
            padding_mask = causal_mask
        
        # 第一層 self-attention：只能看到當前及之前的位置
        attention_output_1 = self.attention_1(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=causal_mask)
        attention_output_1 = self.layernorm_1(inputs + attention_output_1)
        
        # 第二層 cross-attention：從 encoder 輸出取得資訊
        attention_output_2 = self.attention_2(
            query=attention_output_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
        )
        attention_output_2 = self.layernorm_2(
            attention_output_1 + attention_output_2)
        proj_output = self.dense_proj(attention_output_2)
        return self.layernorm_3(attention_output_2 + proj_output)

sequence_length = 100
vocab_size = 15000
embed_dim = 256
latent_dim = 2048
num_heads = 2

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

inputs = keras.Input(shape=(None,), dtype="int64")
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(inputs)
x = TransformerDecoder(embed_dim, latent_dim, num_heads)(x, x)
outputs = layers.Dense(vocab_size, activation="softmax")(x)
model = keras.Model(inputs, outputs)
model.compile(loss="sparse_categorical_crossentropy", optimizer="rmsprop")

tokens_index = dict(enumerate(text_vectorization.get_vocabulary()))
prompt = "This movie"
text_gen_callback = TextGenerator(
    prompt,
    generate_length=50,
    model_input_length=sequence_length,
    temperatures=(0.2, 0.5, 0.7, 1., 1.5))

model.fit(lm_dataset, epochs=200, callbacks=[text_gen_callback])