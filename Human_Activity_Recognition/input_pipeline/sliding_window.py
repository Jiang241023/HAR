import tensorflow as tf

def sliding_window(data, labels, window_size=128, overlap = 0.5):
    # Convert data and labels into tf.data.Dataset object
    step_size = int(window_size*(1 - overlap))

    data_ds = tf.data.Dataset.from_tensor_slices(data)
    labels_ds = tf.data.Dataset.from_tensor_slices(labels)

    # Combine data and label
    dataset = tf.data.Dataset.zip((data_ds, labels_ds))

    # Create the sliding window
    dataset = dataset.window(size=window_size, shift=step_size, drop_remainder=True)

    # Flatten the dataset and batch into windows
    def create_windowed_dataset(data_ds, labels_ds):
        # Batch the windowed data and labels, return as zipped Dataset
        windowed_data = data_ds.batch(window_size, drop_remainder=True)
        windowed_labels = labels_ds.batch(window_size, drop_remainder=True)
        return tf.data.Dataset.zip((windowed_data, windowed_labels))

    # Process each element of the nested data using the create_windowed_dataset function
    dataset = dataset.flat_map(create_windowed_dataset)

    # Calculate the most frequent label
    def most_frequent_labels(window_labels):
        unique_labels, _, counts = tf.unique_with_counts(window_labels)
        most_freq = unique_labels[tf.argmax(counts)]
        return most_freq

    dataset = dataset.map(
        lambda window_data, window_labels: (window_data, most_frequent_labels(window_labels))
    )

    return dataset
