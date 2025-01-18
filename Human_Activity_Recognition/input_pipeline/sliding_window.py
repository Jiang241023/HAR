import tensorflow as tf

def sliding_window(dataset, window_size =128, overlap = 0.5, primary_threshold=0.8 , transition_threshold = 0.3 , labeling_mode="S2L"):

    # Convert data and labels into tf.data.Dataset object
    step_size = int(window_size*(1 - overlap))

    # Create the sliding window -> returns each window as dataset objects containing elements in that window
    dataset = dataset.window(size=window_size, shift=step_size, drop_remainder=True)

    # Pair data and laberls within each window
    def process_window(data_window, label_window):
        data_window = data_window.batch(window_size, drop_remainder=True)
        label_window = label_window.batch(window_size, drop_remainder=True)
        return tf.data.Dataset.zip((data_window, label_window))

    # assign label
    def assign_label(data_window, label_window):
        unique_labels, _, counts = tf.unique_with_counts(label_window)

        # Handle empty window edge case
        if tf.size(unique_labels) == 0:
            return data_window, tf.constant(0, dtype=tf.int32)

        # Calculate dominant label and its proportion
        dominant_label = tf.cast(unique_labels[tf.argmax(counts)], tf.int32)  # Ensure int32 dtype
        dominant_count = tf.reduce_max(counts)
        proportion = tf.cast(dominant_count, tf.float32) / tf.size(label_window, out_type=tf.float32)

        # Assign zero for ambiguous windows
        if dominant_label < 7 and proportion < primary_threshold:
            return data_window, tf.constant(0, dtype=tf.int32)
        elif dominant_label >= 7 and proportion < transition_threshold:
            return data_window, tf.constant(0, dtype=tf.int32)
        else:
            return data_window, dominant_label

    # Apply label assignment and flatten the dataset to create the window tensor
    dataset = dataset.flat_map(process_window)

    # Choose between S2L and S2S labeling
    if labeling_mode == "S2L":
        dataset = dataset.map(assign_label)
    elif labeling_mode == "S2S":
        dataset = dataset.map(lambda data_window, label_window: (data_window, label_window)) # Keep full label sequence
    else:
        raise ValueError("Invalid labeling mode. Choose 'S2L' or 'S2S'.")

    return dataset

