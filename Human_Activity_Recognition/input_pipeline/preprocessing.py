import gin
import tensorflow as tf



@gin.configurable
def preprocess(data):
    """Dataset preprocessing: Normalizing"""
    mean = tf.reduce_mean(data, axis = 0)
    std = tf.math.reduce_std(data, axis = 0)

    return data

augmentation_layer = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(factor=0.2),  # Approximately ±10 degrees,
        tf.keras.layers.RandomZoom(height_factor=0.2, width_factor=0.2), # small zoom
        tf.keras.layers.RandomBrightness(0.1),
        tf.keras.layers.RandomContrast(0.1),
        tf.keras.layers.RandomFlip("horizontal_and_vertical")
    ])

def augment(image, label):
    """Data augmentation"""
   # tf.print("Before augment - Image Min:", tf.reduce_min(image), ", Max:", tf.reduce_max(image))
    image = augmentation_layer(image)
   # tf.print("After augment - Image Min:", tf.reduce_min(image), ", Max:", tf.reduce_max(image))
    return image, label









