import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

np.random.seed(42)  # Make sure that we can get the same color every time

def visualize_data(dataset=None, oversample = False):

    if oversample:
        pass
    else:
        data_path = r"E:\DL_LAB_HAPT_DATASET\HAPT_Data_Set\RawData"
        all_files = r"E:\DL_LAB_HAPT_DATASET\HAPT_Data_Set"

        acc_file = os.path.join(data_path, "acc_exp01_user01.txt")
        acc_data = np.loadtxt(acc_file)
        print("Accelerometer data shape:", acc_data.shape)

        label_files = os.path.join(data_path, "labels.txt")
        labels = np.loadtxt(label_files, dtype=int)
        print(f"labels:\n {labels}")
        print("label shape:", labels.shape)

        exp_id = 1
        labels_exp1 = labels[labels[:, 0] == exp_id] # labels[:, 0] extracts the 0th column
        print(f"labels_exp1:\n {labels_exp1}")
        print(f"labels_exp1 shape={labels_exp1.shape}")

        # Create the full path for activity_labels.txt
        activity_labels_path = os.path.join(all_files, "activity_labels.txt")
        activity_labels = []
        with open(activity_labels_path, "r") as f: # Open the activity in read mode
            # " 1 WALKING DOWNSTAIRS " -> ["1", "WALKING", "DOWNSTAIRS"]
            for line in f:
                parts = line.strip().split() # strip() removes leading and trailing whitespace (spaces, tabs, newlines); split() then splits the stripped string by whitespace into a list of substrings
                print(f"parts: {parts}")
                activity_labels.append((int(parts[0]), " ".join(parts[1:])))

        # Convert list of tuples into a dictionary
        activity_dict = dict(activity_labels)
        print(f"act dict :{activity_dict}")

        time_steps = np.arange(acc_data.shape[0])  # 0 ~ T-1
        print(f"time_steps:{time_steps}")

        # Extracts unique act_id from the third column of labels_exp1 (index 2)
        unique_acts = np.unique(labels_exp1[:, 2])
        cmap = plt.get_cmap("tab20", len(unique_acts))
        color_map = {}
        for i, act_id in enumerate(unique_acts):
            color_map[act_id] = cmap(i)
        # print(f"color_map: {color_map}")

        fig = plt.figure(figsize=(18, 6))

        # Top plot is for accelerometer data
        acc_plot = fig.add_subplot(2,1,1)

        # Iterate over all rows (segments), plotting one small segment at a time.
        for row in labels_exp1:
            _, _, act_id, start_idx, end_idx = row

            # Get the acc_data
            x_segment = acc_data[start_idx:end_idx, 0]
            y_segment = acc_data[start_idx:end_idx, 1]
            z_segment = acc_data[start_idx:end_idx, 2]
            #print(f"x_segment shape: {x_segment.shape}")
            t_segment = time_steps[start_idx:end_idx]

            # Use different colors to distinguish activities
            c = color_map[act_id]

            acc_plot.plot(t_segment, x_segment, color="Green")
            acc_plot.plot(t_segment, y_segment, color="Blue")
            acc_plot.plot(t_segment, z_segment, color="Red")
            acc_plot.axvspan(start_idx, end_idx, color=c, alpha=0.3)

        acc_plot.set_title("Accelerometer for exp01 (all activities) without white space")

        # Bottom plot is for the bar
        bar_plot = fig.add_subplot(2, 1, 2)

        num_acts = len(unique_acts)
        fraction_block_width = 1.0 / num_acts  # fraction of the subplot width per activity

        for i, act_id in enumerate(unique_acts):
            act_name = activity_dict.get(i+1)
            print(f"act_name:{act_name}")
            c = color_map[act_id]

            # Create a rectangle patch for each activity block
            rect = patches.Rectangle(
                (i * fraction_block_width, 0),  # Along the x-axis, the blocks are placed sequentially,
                                                    # with each block shifted by i * fraction_block_width from x = 0.
                                                    # The 0 indicates the rectangle starts at the very bottom of the subplot’s y-axis.
                fraction_block_width,  # width
                1.0,  # height (entire subplot)
                facecolor=c,
                edgecolor="black",
                alpha=0.3 # transparency
            )
            bar_plot.add_patch(rect)

            # Place the text label in the middle of the block
            bar_plot.text(
                (i + 0.5) * fraction_block_width, 0.5, act_name,
                ha="center", va="center", rotation=45, fontsize=8
            )# ha: horizotal alignment; va: vertical alignment

        bar_plot.set_xticks([])
        bar_plot.set_yticks([])

        # Adjust subplot spacing so the text at bottom is visible
        plt.subplots_adjust(hspace=0.8, bottom=0.1)
        plt.show()
if __name__ == "__main__":

    visualize_data(oversample = False)

