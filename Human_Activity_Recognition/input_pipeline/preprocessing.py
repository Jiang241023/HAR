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

def oversample_and_augment(data,label, minority_classes=None, oversample_factor = 2):
    # seperate data into minority and majority class
    minority_indices = tf.where(tf.reduce_any([labels == c for c in minority_classes], axis = 0))
    majority_indices = tf.where(~tf.reduce_any([labels == c for c in minority_classes], axis = 0))

    minority_data = tf.gather(data, minority_indices)
    minority_labels = tf.gather(label, minority_indices)

    majority_data = tf.gather(data, majority_indices)
    majority_labels = tf.gather(data,majority_indices)

    # Repeat and augment the minority class samples
    oversampled_data = tf.repeat(minority_data, repeats= oversample_factor, axis = 0)
    oversampled_labels = tf.repeat(minority_labels, repeats = oversample_factor , axis = 0)

    def augment_ov(data, label):
        jitter = tf.random.uniform(data.shape, minval=-0.1, maxval=0.1)
        scaled_data = data + jitter
        scale_factor = tf.random.uniform([], minval=0.9,maxval=1.1)
        scaled_data = scaled_data * scale_factor
        return scaled_data, label

    oversampled_data , oversampled_labels = tf.map_fn(lambda x: augment(x[0], x[1]),(oversampled_data, oversampled_labels),
                                                      fn_output_signature =(tf.TensorSpec(shape=data.shape[1:], dtype=data.dtype),
                                                                            tf.TensorSpec(shape=label.shape[1:], dtype=label.dtype)))
    combined_data = tf.concat([majority_data, oversampled_data], axis =0)
    combined_labels = tf.concat([majority_labels, oversampled_labels], axis=0)

    # Sort the combined dataset by temporal order
    sorted_indices = tf.argsort(tf.range(tf.shape(combined_data)[0]))
    combined_data = tf.gather(combined_data, sorted_indices)
    combined_labels = tf.gather(combined_labels)

    return combined_data, combined_labels






