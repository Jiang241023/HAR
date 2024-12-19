import gin
import tensorflow as tf

@gin.configurable
def preprocess(data):
    """Dataset preprocessing: Normalizing"""
    mean = tf.reduce_mean(data, axis = 0)
    std = tf.math.reduce_std(data, axis = 0)

    return (data - mean)/ std


def augment(data, label):
    jitter = tf.random.uniform(data.shape, minval=-0.1, maxval=0.1)
    scaled_data = data + jitter
    scale_factor = tf.random.uniform([], minval=0.9, maxval=1.1)
    scaled_data = scaled_data * scale_factor
    return scaled_data, label








