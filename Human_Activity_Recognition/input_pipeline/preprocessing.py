import gin
import tensorflow as tf

@gin.configurable
def preprocess(data):
    """Dataset preprocessing: Normalizing"""
    mean = tf.reduce_mean(data, axis = 0)
    std = tf.math.reduce_std(data, axis = 0)

    return (data - mean)/ std


def augment(data, label):
    # jitter = tf.random.uniform(data.shape, minval=-0.1, maxval=0.1)
    # scaled_data = data + jitter
    scaled_data = data
    # scale_factor = tf.random.uniform([], minval=0.9, maxval=1.1)
    # scaled_data = scaled_data * scale_factor
    return scaled_data, label

def oversample(data, labels, debug=True):
    # ensure labels are 1D
    labels = tf.squeeze(labels)
    # get unique activities and their counts
    activities, _,activity_counts = tf.unique_with_counts(labels) # unique values, indices of unique values, counts
    primary_mask = activities < 7

    max_activity = tf.reduce_max(tf.boolean_mask(activity_counts,primary_mask))
    max_transition_activity = tf.reduce_max(tf.boolean_mask(activity_counts,~primary_mask))

    activities_in_order = tf.unique(labels)[0]

    # Initialize list for oversampled data and labels
    oversampled_data = []
    oversampled_labels = []

    # Oversample each activity
    for activity in activities_in_order:
        activity_indices = tf.where(labels == activity)[:,0]

        # Determine oversampling size
        if activity < 7 :
            oversample_size = max_activity
        else:
            oversample_size = max_transition_activity

        # Debug info
        if debug:
            print(f"Activity: {activity}")
            print(f"Original count: {len(activity_indices)}")
            print(f"Oversample size: {oversample_size}")

        # Perform random sampling with replacement
        sampled_indices = tf.random.uniform([oversample_size],minval=0, maxval=tf.shape(activity_indices)[0], dtype=tf.int32)
        sampled_indices = tf.gather(activity_indices, sampled_indices)

        # Gather oversampled data and labels
        oversampled_data.append(tf.gather(data, sampled_indices))
        oversampled_labels.append(tf.gather(labels, sampled_indices))

    # Concatenate oversampled data and labels
    oversampled_data = tf.concat(oversampled_data, axis=0)
    oversampled_labels = tf.concat(oversampled_labels, axis=0)

    dataset = tf.data.Dataset.from_tensor_slices((oversampled_data,oversampled_labels))

    return dataset






