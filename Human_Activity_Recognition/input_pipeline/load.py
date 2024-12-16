import tensorflow as tf
import numpy as np
import os

data_dir = r'E:\DL_LAB_HAPT_DATASET\HAPT_Data_Set\RawData'
labels_file = r'E:\DL_LAB_HAPT_DATASET\HAPT Data Set\RawData\labels.txt'


def load(data_dir = data_dir):

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

def normalize_data(data):
    mean = tf.reduce_mean(data, axis=0)
    std = tf.math.reduce_std(data, axis=0)
    return (data - mean)/ std

def cal_exp_lengths(data_path, file_prefix ='acc_exp', file_suffix=".txt"):
    exp_length =[0]
    filenames = sorted(os.listdir(data_path))
    for file in filenames:
        if file.startswith(file_prefix) and file.endswith(file_suffix):
            file_path = os.path.join(data_path,file)
            num_rows = np.loadtxt(file_path).shape[0]
            exp_length.append(num_rows)
    return exp_length


def parse_labels(label_file_path, exp_lengths):
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
    label_tensor = np.zeros(total_time_steps, dtype=int)
    for segment in segments:
      activity_id = segment["activity_id"]
      start_idx = segment["start_idx"]
      end_idx = segment["end_idx"]

      # Update label_array
      label_tensor[start_idx:end_idx] = activity_id
    return label_tensor

def sliding_window(data, labels, window_size =128, step_size = 64):
    # Convert data and labels into tf.data.Dateset object
    data_ds = tf.data.Dataset.from_tensor_slices(data)
    labels_ds = tf.data.Dataset.from_tensor_slices(labels)

    # Combine data and label
    dataset = tf.data.Dataset.zip((data_ds, labels_ds))
    #Create the sliding window
    dataset = dataset.window(size=window_size, shift =step_size, drop_remainder=True)

    # Flatten the dataset and batch into windows
    dataset = dataset.flat_map(lambda window_data,window_labels : tf.data.Dataset.zip( window_data.batch(window_size), window_labels.batch(window_size)))

    # Calcualte the most frequent label
    def most_frequent_labels(window_labels):
        unique_labels , _ , counts = tf.unique_with_counts(window_labels)
        most_freq = unique_labels[tf.argmax(counts)]
        return most_freq
    dataset = dataset.map(lambda window_data, window_labels : (window_data,most_frequent_labels(window_labels)))

    return dataset

ds_train, ds_val, ds_test = load(data_dir)
norm_data_ds_train = normalize_data(ds_train)
norm_data_ds_val = normalize_data(ds_val)
norm_data_ds_test = normalize_data(ds_test)

experiment_length = cal_exp_lengths(data_dir)
total_time_steps = sum(experiment_length)
segments = parse_labels(labels_file, exp_lengths=experiment_length)
print("First few segmens: ,", segments[:50])
label_tensor = create_label_tensor(segments, total_time_steps)
print("Label tensor shape :", label_tensor.shape)
print("Fist 50 labels",label_tensor[250:300])
sliding_window_data_ds_train = sliding_window(ds_train, label_tensor, window_size=128, step_size=64)
# sliding_window_data_ds_val = sliding_window(ds_val , label_tensor, window_size=128, step_size=64)
# sliding_window_data_ds_test = sliding_window(ds_test, label_tensor, window_size=128, step_size=64)

for window_data, window_labels in sliding_window_data_ds_train.take(10):
    print("Window Data Shape: ", window_data.shape)
    print("Window Labels Shape :", window_labels.shape)
    print("Window Labels : ",window_labels.numpy())
    print("="*50)

# for window_data, window_labels in sliding_window_data_ds_val.take(10):
#     print("Window Data Shape: ", window_data.shape)
#     print("Window Labels Shape :", window_labels.shape)
#     print("Window Labels : ",window_labels.numpy())
#     print("="*50)
#
# for window_data, window_labels in sliding_window_data_ds_test.take(10):
#     print("Window Data Shape: ", window_data.shape)
#     print("Window Labels Shape :", window_labels.shape)
#     print("Window Labels : ",window_labels.numpy())
#     print("="*50)