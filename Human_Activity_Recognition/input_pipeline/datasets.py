import gin
from tensorflow.data.experimental import AUTOTUNE
import tensorflow as tf
import logging
from input_pipeline.preprocessing import preprocess
import tensorflow_datasets as tfds
from tensorflow.keras.utils import image_dataset_from_directory
import numpy as np
import os

def cal_exp_lengths(data_path, file_prefix ='acc_exp', file_suffix=".txt"):
    """
    Calculate the number of rows for each experiment file and return the cumulative lengths.
    """
    exp_length =[0]
    filenames = sorted(os.listdir(data_path))
    for file in filenames:
        if file.startswith(file_prefix) and file.endswith(file_suffix):
            file_path = os.path.join(data_path, file)
            num_rows = np.loadtxt(file_path).shape[0]
            exp_length.append(num_rows)
    return exp_length

def parse_labels(label_file_path, exp_lengths):
    """
    Parse the label file and adjust the activity start and end indices based on cumulative experiment lengths.
    """
    segments =[]
    cumulative_offset = 0
    current_experiment_id = -1
    with open(label_file_path,'r') as f:
        for line in f:
            parts = line.strip().split()
            exp_id = int(parts[0])
            act_id = int(parts[2])
            start_idx = int(parts[3])
            end_idx = int(parts[4])

            # Update the cummulative offset for next experiment
            if exp_id != current_experiment_id:
                cumulative_offset += exp_lengths[exp_id -1] # Add previous exp length , intial element in exp_len = 0
                current_experiment_id = exp_id

            adjusted_start_idx = start_idx + cumulative_offset
            adjusted_end_idx = end_idx + cumulative_offset

            segments.append({
                "activity_id" : act_id,
                "start_idx" : adjusted_start_idx,
                "end_idx" : adjusted_end_idx
            })
    return segments

def create_label_tensor(segments, total_time_steps):
    """
    Generate a label tensor for all time steps based on parsed activity segments.
    """
    label_tensor = np.zeros(total_time_steps, dtype=int)
    for segment in segments:
      activity_id = segment["activity_id"]
      start_idx = segment["start_idx"]
      end_idx = segment["end_idx"]

      # Update label_array
      label_tensor[start_idx:end_idx] = activity_id
    return label_tensor

def sliding_window(data, labels, window_size=128, overlap = 0.5):

    # Convert data and labels into tf.data.Dataset object
    step_size = int(window_size*(1 - overlap))

    data_ds = data
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

@gin.configurable
def load(name, data_dir, labels_file):
    if name == "HAPT":
        logging.info(f"Preparing dataset {name}...")

        acc_files = sorted([f for f in os.listdir(data_dir) if f.startswith("acc")])
        gyro_files = sorted([f for f in os.listdir(data_dir) if f.startswith("gyro")])

        # combined_data = []
        print(f"acc_files :{acc_files}")
        print(f"gyro_files :{gyro_files}")
        print("Matching and concatenating...")

        ds_train = []
        ds_val = []
        ds_test = []

        for acc_file, gyro_file in zip(acc_files, gyro_files):

            acc_file_path = os.path.join(data_dir, acc_file)
            gyro_file_path = os.path.join(data_dir, gyro_file)
            acc_data = tf.convert_to_tensor(np.loadtxt(acc_file_path), dtype=tf.float32)
            gyro_data = tf.convert_to_tensor(np.loadtxt(gyro_file_path), dtype=tf.float32)
            combined = tf.concat([acc_data, gyro_data], axis=1)

            # Extract user ID from filename
            user_id = int(acc_file.split('_user')[1].split('.txt')[0])

            if 1 <= user_id <= 21:
                ds_train.append(combined)
            elif 28 <= user_id <= 30:
                ds_val.append(combined)
            elif 22 <= user_id <= 27:
                ds_test.append(combined)

        ds_train = tf.concat(ds_train, axis=0)  # aixs = 0 means vertical concatenation
        print(f'Completed.\nds_train shape: {ds_train.shape}')

        ds_val = tf.concat(ds_val, axis=0)
        print(f'Completed.\nds_val shape: {ds_val.shape}')

        ds_test = tf.concat(ds_test, axis=0)
        print(f'Completed.\nds_test shape: {ds_test.shape}')

    else:
        raise ValueError

    experiment_length = cal_exp_lengths(data_dir)
    total_time_steps = sum(experiment_length)
    segments = parse_labels(labels_file, exp_lengths=experiment_length)
    print("First few segmens: ,", segments[:50])
    label_tensor = create_label_tensor(segments, total_time_steps)
    print("Label tensor shape :", label_tensor.shape)
    print("Fist 50 labels", label_tensor[250:300])

    # Calculate cumulative lengths for splitting
    train_length = ds_train.shape[0]
    val_length = ds_val.shape[0]

    # Split label_tensor
    train_labels = label_tensor[:train_length]
    val_labels = label_tensor[train_length:train_length + val_length]
    test_labels = label_tensor[train_length + val_length:]

    # Prepare
    ds_train, ds_val, ds_test = prepare(ds_train, ds_val, ds_test, train_labels, val_labels, test_labels)

    return ds_train, ds_val, ds_test


@gin.configurable
def prepare(ds_train, ds_val, ds_test, train_labels, val_labels, test_labels, ds_info=None, caching=True):
    """Prepare datasets with preprocessing, batching, caching, and prefetching"""
    ds_train = tf.data.Dataset.from_tensor_slices(ds_train)
    ds_val = tf.data.Dataset.from_tensor_slices(ds_val)
    ds_test = tf.data.Dataset.from_tensor_slices(ds_test)

    # Prepare training dataset
    ds_train = ds_train.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = sliding_window(ds_train, train_labels, window_size=128, overlap=0.5)
    if caching:
        ds_train = ds_train.cache()

    if ds_info:
       shuffle_buffer_size = ds_info.get("num_examples", 1000) // 10  # Default to 1000 if ds_info not provided
       ds_train = ds_train.shuffle(shuffle_buffer_size)
    else:
        ds_train = ds_train.shuffle(1000)  # Fallback shuffle size

    ds_train = ds_train.repeat().prefetch(tf.data.AUTOTUNE)

    # Prepare validation dataset (no augmentation)
    ds_val = ds_val.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds_val = sliding_window(ds_val, val_labels, window_size=128, overlap=0.5)
    if caching:
        ds_val = ds_val.cache()
    ds_val = ds_val.prefetch(tf.data.AUTOTUNE)

    # Prepare test dataset if available
    if ds_test is not None:
        ds_test = ds_test.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        ds_test = sliding_window(ds_test, test_labels, window_size=128, overlap=0.5)
        if caching:
            ds_test = ds_test.cache()
        ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_val, ds_test

data_dir = r'E:\DL_LAB_HAPT_DATASET\HAPT_Data_Set\RawData'
labels_file = r'E:\DL_LAB_HAPT_DATASET\HAPT_Data_Set\RawData\labels.txt'
name = "HAPT"

ds_train, ds_val, ds_test = load(name, data_dir, labels_file)

datasets = [
    ("train", ds_train),
    ("val", ds_val),
    ("test", ds_test)
    ]

for name, dataset in datasets:
    print(f"Processing the dataset of {name}...")
    for window_data, window_labels in dataset.take(10):
        print("Window Data Shape: ", window_data.shape)
        print("Window Labels Shape :", window_labels.shape)
        print("Window Labels : ",window_labels.numpy())
        print("="*50)

