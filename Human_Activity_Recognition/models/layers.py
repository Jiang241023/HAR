import gin
import tensorflow as tf
from tensorflow.keras.regularizers import l2


@gin.configurable
def vgg_block(inputs, filters, kernel_size):
    """A single VGG block consisting of two convolutional layers, followed by a max-pooling layer.
    Parameters:
        inputs (Tensor): input of the VGG block
        filters (int): number of filters used for the convolutional layers
        kernel_size (tuple: 2): kernel size used for the convolutional layers, e.g. (3, 3)
    Returns:
        (Tensor): output of the VGG block
    """

    out = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', activation = 'relu', kernel_regularizer=l2(1e-4))(inputs)
    out = tf.keras.layers.MaxPool2D((2, 2))(out)

    out = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', activation = 'relu', kernel_regularizer=l2(1e-4))(out)
    out = tf.keras.layers.MaxPool2D((2, 2))(out)

    return out

@gin.configurable
def lstm_block(inputs, lstm_units, dense_units, dropout_rate = 0.5):

    # LSTM Layer