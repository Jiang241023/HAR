#import gin
import tensorflow as tf
from tensorflow.keras.regularizers import l2


def lstm_block(inputs, lstm_units, dropout_rate):

    # LSTM Layer
    out = tf.keras.layers.LSTM(lstm_units, return_sequences = True, kernel_regularizer = l2(1e-4))(inputs)
    out = tf.keras.layers.Dropout(dropout_rate)(out)

    return out

#@gin.configurable
def gru_block(inputs, gru_units, dropout_rate):

    # LSTM Layer
    out = tf.keras.layers.GRU(gru_units, return_sequences = True, kernel_regularizer = l2(1e-4))(inputs)
    out = tf.keras.layers.Dropout(dropout_rate)(out)

    return out

#@gin.configurable
def transformer_block(inputs, num_heads, ff_dim, dropout_rate):

    # Multi-Head Attention
    attn_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1])(inputs, inputs)
    attn_output = tf.keras.layers.Dropout(dropout_rate)(attn_output)
    out1 = tf.keras.layers.Add()([inputs, attn_output])
    out1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(out1)

    # Feedforward network
    ffn_output = tf.keras.layers.Dense(ff_dim, activation='relu')(out1)
    ffn_output = tf.keras.layers.Dense(inputs.shape[-1])(ffn_output)
    ffn_output = tf.keras.layers.Dropout(dropout_rate)(ffn_output)
    out2 = tf.keras.layers.Add()([out1, ffn_output])
    out2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(out2)

    return out2