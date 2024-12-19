import gin
import tensorflow as tf

@gin.configurable
def preprocess(data):
    """Dataset preprocessing: Normalizing"""
    mean = tf.reduce_mean(data, axis = 0)
    std = tf.math.reduce_std(data, axis = 0)

    return (data - mean)/ std











