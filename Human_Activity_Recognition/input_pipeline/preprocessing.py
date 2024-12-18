import gin
import tensorflow as tf



@gin.configurable
def preprocess(data):
    """Dataset preprocessing: Normalizing"""
    mean = tf.reduce_mean(data, axis = 0)
    std = tf.math.reduce_std(data, axis = 0)

    return (data - mean)/ std



def augment(image, label):
    """Data augmentation"""
   # tf.print("Before augment - Image Min:", tf.reduce_min(image), ", Max:", tf.reduce_max(image))
    #image = augmentation_layer(image)
   # tf.print("After augment - Image Min:", tf.reduce_min(image), ", Max:", tf.reduce_max(image))
    return image, label









