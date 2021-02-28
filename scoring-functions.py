import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math

class AdditiveAttention(tf.Module):
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = tf.keras.layers.Dense(num_hiddens, activation=None)
        self.W_q = tf.keras.layers.Dense(num_hiddens, activation=None)
        self.w_v = tf.keras.layers.Dense(1, activation=None)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def __call__(self, queries, keys, values):
        queries, keys = self.W_q(queries), self.W_k(keys)
        features = tf.expand_dims(queries, axis=2) + tf.expand_dims(keys, axis=1)
        features = tf.keras.activations.tanh(features)
        scores = tf.squeeze(self.w_v(features), axis=-1)
        self.attention_weights = tf.nn.softmax(scores)
        return tf.matmul(self.dropout(self.attention_weights), values)

# example run through
queries, keys = tf.random.normal((2,1,20), mean=0.0, stddev=1), tf.random.uniform((2, 10, 2))
values = tf.repeat(tf.reshape(tf.range(0, 40, dtype=tf.float32), (1, 10, 4)), 2, axis=0)
attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8, dropout=0.1)
result = attention(queries, keys, values)
print(result)

class ScaledDotProductAttention(tf.Module):
    def __init__(self, dropout, **kwargs):
        super(ScaledDotProductAttention, self).__init__(**kwargs)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def __call__(self, queries, keys, values):
        d = queries.shape[-1]
        scores = tf.matmul(queries, keys, transpose_b = True)
        self.attention_weights = tf.nn.softmax(scores)
        return tf.matmul(self.dropout(self.attention_weights), values)

queries = tf.random.normal((2,1,2), 0.0, 1.0)
attention = ScaledDotProductAttention(dropout=0.5)
result = attention(queries, keys, values)
print(result)
