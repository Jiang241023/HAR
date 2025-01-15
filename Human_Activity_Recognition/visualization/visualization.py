import os
import numpy as np
import matplotlib.pyplot as plt
def visualize_data(dataset=None, oversample = False):

    if oversample:
        pass
    else:
        data_path = r"E:\DL_LAB_HAPT_DATASET\HAPT_Data_Set\RawData"
        all_files = r"E:\DL_LAB_HAPT_DATASET\HAPT_Data_Set"

        acc_file = os.path.join(data_path, "acc_exp01_user01.txt")
        acc_data = np.loadtxt(acc_file)

        label_files = os.path.join(data_path, "labels.txt")
        labels = np.loadtxt(label_files, dtype=int)

        print("label shape:", labels.shape)
        print("Accelerometer data shape:", acc_data.shape)

        # Create the full path for activity_labels.txt
        activity_labels_path = os.path.join(all_files, "activity_labels.txt")
        activity_labels = []
        with open(activity_labels_path, "r") as f: # Open the activity in read mode
            for line in f:
                parts = line.strip().split() # strip() removes leading and trailing whitespace (spaces, tabs, newlines);
                activity_labels.append((int(parts[0]), " ".join(parts[1:])))
        activity_dict = dict(activity_labels)
        print(f"act dict :{activity_dict}")

        exp_id = 1
        labels_exp1 = labels[labels[:, 0] == exp_id]

        print(f"acc_data shape={acc_data.shape}")
        print(f"labels_exp1 shape={labels_exp1.shape}")

        time_steps = np.arange(acc_data.shape[0])  # 0 ~ T-1

        # 为所有出现的 activity_id 指定不同颜色
        unique_acts = np.unique(labels_exp1[:, 2])
        color_map = {}
        np.random.seed(123)  # 固定随机种子，保证每次颜色一致
        for act_id in unique_acts:
            color = "#%06x" % np.random.randint(0, 0xFFFFFF)
            color_map[act_id] = color

        plt.figure(figsize=(12, 6))

        # 遍历所有行(片段)，一次绘制一小段
        for row in labels_exp1:
            _, _, act_id, start_idx, end_idx = row
            # 选择在这里只画X轴数据
            x_segment = acc_data[start_idx:end_idx, 0]
            t_segment = time_steps[start_idx:end_idx]

            # 用不同颜色区分活动
            c = color_map[act_id]
            act_name = activity_dict.get(act_id, f"Act{act_id}")

            plt.plot(t_segment, x_segment, color=c, label=act_name)

        # 注意：不同片段如果活动相同，会重复出图例label
        # 可以在后面手动设置图例
        plt.title("Accelerometer X for exp01 (all activities)")
        plt.xlabel("Timesteps")
        plt.ylabel("Acceleration (X)")

        # 去重图例
        handles, labels_ = plt.gca().get_legend_handles_labels()
        unique = dict(zip(labels_, handles))  # 字典会保留最后一次label对应的handle
        plt.legend(unique.values(), unique.keys(), loc="upper right")

        plt.grid()
        plt.show()
if __name__ == "__main__":

    visualize_data(oversample = False)

