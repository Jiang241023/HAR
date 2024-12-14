import tensorflow as tf
import pandas as pd
import numpy as np
import os

data_dir = r'E:\DL_LAB_HAPT_DATASET\HAPT_Data_Set\RawData'
labels_file = r'E:\DL_LAB_HAPT_DATASET\HAPT_Data_Set\RawData\labels.txt'

def load(data_dir = data_dir , labels_file = labels_file):

    acc_files = sorted([f for f in os.listdir(data_dir) if f.startswith("acc")])
    gyro_files = sorted([f for f in os.listdir(data_dir) if f.startswith("gyro")])

    #combined_data = []
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
        combined = tf.concat([acc_data, gyro_data], axis = 1)

        # Extract user ID from filename
        user_id = int(acc_file.split('_user')[1].split('.txt')[0])

        if 1 <= user_id <= 21:
            ds_train.append(combined)
        elif 28 <= user_id <= 30:
            ds_val.append(combined)
        elif 22 <= user_id <= 27:
            ds_test.append(combined)


    ds_train = tf.concat(ds_train, axis = 0) # aixs = 0 means vertical concatenation
    print(f'Completed.\nCombined shape:{ds_train.shape}')

    ds_val = tf.concat(ds_val, axis = 0)
    print(f'Completed.\nCombined shape:{ds_val.shape}')

    ds_test = tf.concat(ds_test, axis = 0)
    print(f'Completed.\nCombined shape:{ds_test.shape}')


    return ds_train, ds_val, ds_test

def sliding_window(window_length = 256, overlap = 0.5, data = data, labels = labels):

    # Convert data and labels into tf.data.Dataset object
    data_ds = tf.data.Dataset.from_tensor_slices(data)
    labels_ds = tf.data.Dataset.from_tensor_slices(labels)
    dataset = tf.data.Dataset.zip((data_ds, labels_ds))

    # Create the sliding window
    window_shift = int(window_length * (1 - overlap))
    dataset = dataset.window(size = window_length, shift = window_shift, drop_remainder = True)

    # Flatten the dataset and batch into windows
    dataset = dataset.flat_map(lambda window_data, window_labels: tf.data.Dataset.zip(window_data.batch(window_length), window_labels.batch(window_length)))

    return dataset

ds_train, ds_val, ds_test = load(window_length = 128, overlap = 0.5, data_dir = data_dir , labels_file = labels_file)
print(ds_train)
print(ds_val)
print(ds_test)