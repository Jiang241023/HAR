import gin
import tensorflow as tf
from tensorflow.keras.applications import MobileNet, VGG16, InceptionResNetV2
from models.layers import lstm_block, gru_block, transformer_block
from tensorflow.keras.regularizers import l2

@gin.configurable
def lstm_like(n_classes, lstm_units, dense_units, n_blocks, dropout_rate, input_shape = (128, 6)):

    #Load the pretrained VGG16 model excluding the top classification layer
    assert n_blocks > 0
    inputs = tf.keras.Input(shape = input_shape)
    x = inputs
    print(f"input shape:{x.shape}")
    for _ in range(n_blocks - 1):
        x = lstm_block(x)
    lstm_out = tf.keras.layers.LSTM(lstm_units, return_sequences=True, kernel_regularizer=l2(1e-4))(x)
    print(f"lstm_out when return_sequences=True shape:{lstm_out.shape}")
    lstm_out = tf.keras.layers.Flatten()(lstm_out)
    x = tf.keras.layers.Dropout(dropout_rate)(lstm_out)
    x = tf.keras.layers.Dense(dense_units, kernel_regularizer=l2(1e-4))(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(n_classes, activation = 'softmax', kernel_regularizer=l2(1e-4))(x)

    return tf.keras.Model(inputs = inputs, outputs=outputs, name='lstm_like')

@gin.configurable
def gru_like(n_classes, gru_units, dense_units, n_blocks, dropout_rate, input_shape = (128, 6)):

    #Load the pretrained VGG16 model excluding the top classification layer
    assert n_blocks > 0
    inputs = tf.keras.Input(shape = input_shape)
    x = inputs
    print(f"input shape:{x.shape}")
    for _ in range(n_blocks - 1):
        x = gru_block(x)
    gru_out = tf.keras.layers.GRU(gru_units, return_sequences=True, kernel_regularizer=l2(1e-4))(x)
    print(f"gru_out when return_sequences=True shape:{gru_out.shape}")
    x = tf.keras.layers.Flatten()(gru_out)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(dense_units, kernel_regularizer=l2(1e-4))(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(n_classes, activation = 'softmax', kernel_regularizer=l2(1e-4))(x)

    return tf.keras.Model(inputs = inputs, outputs=outputs, name='gru_like')


@gin.configurable
def transformer_like(n_classes, dense_units, n_blocks, dropout_rate, input_shape=(128, 6)):

    #Load the pretrained VGG16 model excluding the top classification layer
    assert n_blocks > 0

    inputs = tf.keras.Input(shape=input_shape)
    x = inputs

    for _ in range(n_blocks):
        x = transformer_block(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(dense_units, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(n_classes, activation='softmax', kernel_regularizer=l2(1e-4))(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='transformer_like')
