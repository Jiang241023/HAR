import gin
import tensorflow as tf
from tensorflow.keras.applications import MobileNet, VGG16, InceptionResNetV2
from models.layers import lstm_block
from tensorflow.keras.regularizers import l2

@gin.configurable
def lstm_like(n_classes, lstm_units, dense_units, n_blocks, dropout_rate, input_shape = (128, 6)):

    #Load the pretrained VGG16 model excluding the top classification layer
    assert n_blocks > 0
    out = tf.keras.Input(shape = input_shape)
    print(out.shape)
    if n_blocks >= 1:
        for i in range(n_blocks - 1):
            out = lstm_block(out, n_blocks)
        out = tf.keras.layers.GlobalAveragePooling2D()(out)
        out = tf.keras.layers.Dense(dense_units, kernel_regularizer=l2(1e-4))(out)
        out = tf.keras.layers.LeakyReLU(alpha=0.01)(out)
        out = tf.keras.layers.Dropout(dropout_rate)(out)
        outputs = tf.keras.layers.Dense(n_classes - 1, activation = 'softmax', kernel_regularizer=l2(1e-4))(out)
    else:
        out = lstm_block(out, n_blocks)
        out = tf.keras.layers.GlobalAveragePooling2D()(out)
        out = tf.keras.layers.Dense(dense_units, kernel_regularizer=l2(1e-4))(out)
        out = tf.keras.layers.LeakyReLU(alpha=0.01)(out)
        out = tf.keras.layers.Dropout(dropout_rate)(out)
        outputs = tf.keras.layers.Dense(n_classes - 1, activation='softmax', kernel_regularizer=l2(1e-4))(out)

    return tf.keras.Model(inputs = out, outputs=outputs, name='lstm_like')

