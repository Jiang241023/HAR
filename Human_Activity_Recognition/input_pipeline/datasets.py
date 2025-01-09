import gin
from tensorflow.data.experimental import AUTOTUNE
import tensorflow as tf
import logging
from input_pipeline.preprocessing import preprocess, augment, oversample
from input_pipeline.sliding_window import sliding_window
from input_pipeline.parse_labels import cal_exp_lengths, parse_labels, create_label_tensor
import numpy as np
import os
import time

@gin.configurable
def load(batch_size, name, data_dir, labels_file):
    if name == "HAPT":
        logging.info(f"Preparing dataset {name}...")

        acc_files = sorted([f for f in os.listdir(data_dir) if f.startswith("acc")])
        gyro_files = sorted([f for f in os.listdir(data_dir) if f.startswith("gyro")])

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
        # print(f'Completed.\nds_train shape: {ds_train.shape}')

        ds_val = tf.concat(ds_val, axis=0)
        # print(f'Completed.\nds_val shape: {ds_val.shape}')

        ds_test = tf.concat(ds_test, axis=0)
        # print(f'Completed.\nds_test shape: {ds_test.shape}')

    else:
        raise ValueError

    experiment_length = cal_exp_lengths(data_dir)
    total_time_steps = sum(experiment_length)
    segments = parse_labels(labels_file, exp_lengths=experiment_length)
    label_tensor = create_label_tensor(segments, total_time_steps)

    # Calculate cumulative lengths for splitting
    train_length = ds_train.shape[0]
    val_length = ds_val.shape[0]
    test_length = ds_test.shape[0]
    # print(f"train_length:{train_length}")
    # print(f"val_length:{val_length}")
    # print(f"test_length:{test_length}")
    # Split label_tensor
    train_labels = label_tensor[:train_length]
    test_labels = label_tensor[train_length:train_length+test_length]
    val_labels = label_tensor[train_length + test_length:]
    # print(f"train_labels:{train_labels}")
    # print(f"test_labels:{test_labels}")
    # print(f"val_labels:{val_labels}")
    # Prepare
    ds_train, ds_val, ds_test = prepare(ds_train, ds_val, ds_test, train_labels, val_labels, test_labels, batch_size)

    return ds_train, ds_val, ds_test, batch_size


@gin.configurable
def prepare(ds_train, ds_val, ds_test, train_labels, val_labels, test_labels , batch_size,ds_info=None, caching=True):
    """Prepare datasets with preprocessing, batching, caching, and prefetching"""

    def prepare_dataset(data, labels, batch_size, window_size =128, overlap = 0.5, shuffle_buffer = 1000, cache = True , is_training=True,
                        minority_classes=None, oversample_factor=3, debug=False):
        # Step 1 : Normalize the data
        start_time = time.time()
        print("Normalizing data...")
        data = preprocess(data)
        end_time = time.time()
        print(f"Completed.\nNormalized data shape: {data.shape}")
        print(f"Total time taken to normalize data: {end_time - start_time} seconds")

        if is_training and minority_classes:
            start_time = time.time()
            print("Oversampling and augmenting data...")
            datasets = oversample(data, labels,debug=debug)
            end_time = time.time()
            print(f"Completed.\nOversampled and augmented data shape: {data.shape}")
            print(f"Total time taken to oversample and augment data: {end_time - start_time} seconds")
        else:
            datasets = tf.data.Dataset.from_tensor_slices((data, labels))
        # Step 2 : Create sliding window
        start_time = time.time()
        dataset = sliding_window(datasets, window_size=window_size, overlap=overlap)
        end_time = time.time()
        print(f"Total time taken to create sliding window: {end_time - start_time} seconds")

        # Step 3 : since previous steps are deterministic, caching is done before preprocessing
        if cache:
            dataset = dataset.cache()

        if is_training:
            start_time = time.time()
            print("Augmenting data...")
            dataset = dataset.map(augment , num_parallel_calls=tf.data.AUTOTUNE)
            end_time = time.time()
            print(f"Completed.\nAugmented data shape: {data.shape}")
            print(f"Total time taken to augment data: {end_time - start_time} seconds")

        if is_training:
            dataset = dataset.shuffle(shuffle_buffer).repeat()
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        return dataset

    # Prepare datasets
    ds_train = prepare_dataset(ds_train, train_labels, batch_size, cache= True, minority_classes = minority_classes, debug= True)
    start_time = time.time()
    print("Preparing validation dataset...")
    ds_val = prepare_dataset(ds_val, val_labels, batch_size, cache= True,is_training=False,)
    end_time = time.time()
    print(f"Completed.\nTotal time taken to prepare validation dataset: {end_time - start_time} seconds")
    start_time = time.time()
    print("Preparing test dataset...")
    ds_test = prepare_dataset(ds_test, test_labels, batch_size,cache= True,is_training=False,)
    end_time = time.time()
    print(f"Completed.\nTotal time taken to prepare test dataset: {end_time - start_time} seconds")

    return ds_train, ds_val, ds_test
minority_classes = [7, 8, 9, 10, 11, 12]
# data_dir = r'E:\DL_LAB_HAPT_DATASET\HAPT_Data_Set\RawData'
# labels_file = r'E:\DL_LAB_HAPT_DATASET\HAPT_Data_Set\RawData\labels.txt'
# name = "HAPT"
# batch_size = 64
#
# ds_train, ds_val, ds_test, batch_size = load(batch_size, name, data_dir, labels_file)
#
# datasets = [
#     ("train", ds_train),
#     ("val", ds_val),
#     ("test", ds_test)
#     ]
#
# for name, dataset in datasets:
#     print(f"Processing the dataset of {name}...")
#     for window_data, window_labels in dataset.take(1):
#         print("Window Data Shape: ", window_data.shape)
#         print("Window Labels Shape :", window_labels.shape)
#         print("Window Labels : ",window_labels.numpy())
#         print("="*50)

