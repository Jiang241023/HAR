import gin
import tensorflow as tf
from models.layers import lstm_block, gru_block, transformer_block
from tensorflow.keras.regularizers import l2
from transformers import TFBertModel

@gin.configurable
def lstm_like(n_classes, lstm_units, dense_units, n_blocks, dropout_rate_lstm_block, dropout_rate_dense_layer, input_shape = (128, 6), labeling_mode='S2L'):

    #Load the pretrained VGG16 model excluding the top classification layer
    assert n_blocks > 0
    inputs = tf.keras.Input(shape = input_shape)
    x = inputs
    print(f"input shape:{x.shape}")
    for _ in range(n_blocks - 1):
        x = lstm_block(x, lstm_units, dropout_rate_lstm_block)

    lstm_output = tf.keras.layers.LSTM(lstm_units, return_sequences=True, kernel_regularizer=l2(1e-4))(x)
    print(f"lstm_out when return_sequences=True shape:{x.shape}")

    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(lstm_output)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(lstm_output)
    x = tf.keras.layers.Concatenate()([avg_pool, max_pool])
    x = tf.keras.layers.Dropout(dropout_rate_lstm_block)(x)
    x = tf.keras.layers.Dense(dense_units, kernel_regularizer=l2(1e-4))(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
    x = tf.keras.layers.Dropout(dropout_rate_dense_layer)(x)
    outputs =tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_classes, activation='softmax', kernel_regularizer=l2(1e-4)))(x)

    return tf.keras.Model(inputs = inputs, outputs=outputs, name='lstm_like')

@gin.configurable
def gru_like(n_classes, gru_units, dense_units, n_blocks, dropout_rate, input_shape = (128, 6),labeling_mode='S2L'):

    #Load the pretrained VGG16 model excluding the top classification layer
    assert n_blocks > 0
    inputs = tf.keras.Input(shape = input_shape)
    x = inputs
    print(f"input shape:{x.shape}")
    for _ in range(n_blocks - 1):
        x = gru_block(x)
    gru_out = tf.keras.layers.GRU(gru_units, return_sequences=True,kernel_regularizer=l2(1e-4))(x)

    # # Attention Mechanism
    # attention_scores = tf.keras.layers.Dense(1, activation='tanh')(gru_out)  # Compute alignment scores
    # print(f"attention_scores: {attention_scores}")
    # attention_weights = tf.keras.layers.Softmax(axis=1)(attention_scores)  # Normalize scores
    # print(f"attention_weights: {attention_weights}")
    # context_vector = tf.keras.layers.Multiply()([gru_out, attention_weights])  # Weighted sum of GRU outputs
    # print(f"context_vector (Weighted sum of GRU outputs): {context_vector}")
    # context_vector = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(context_vector)  # Sum along time axis
    # print(f"context_vector (Weighted sum of GRU outputs): {context_vector}")
    # print(f"Context vector shape: {context_vector.shape}")

    if labeling_mode == 'S2L':
        # Pooling layers
        avg_pool = tf.keras.layers.GlobalAveragePooling1D()(gru_out)
        print(f"avg_pool: {avg_pool}")
        max_pool = tf.keras.layers.GlobalMaxPooling1D()(gru_out)
        print(f"max_pool: {max_pool}")
        x = tf.keras.layers.Concatenate()([avg_pool, max_pool])
        print(f"x: {x}")
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        x = tf.keras.layers.Dense(dense_units, kernel_regularizer=l2(1e-4))(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        outputs = tf.keras.layers.Dense(n_classes, activation='softmax', kernel_regularizer=l2(1e-4))(x)
    elif labeling_mode == 'S2S':
        # Dense layers
        x = tf.keras.layers.Dropout(dropout_rate)(gru_out)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(dense_units, kernel_regularizer=l2(1e-4)))(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_classes, activation='softmax', kernel_regularizer=l2(1e-4)))(x)

    return tf.keras.Model(inputs = inputs, outputs=outputs, name='gru_like')
