import numpy as np
import matplotlib.pyplot as plt
import os

from datetime import datetime


def generate_box_plot(timing_dict):
    directory = "Data_Visualizer/data/"
    timestamp = datetime.now().strftime("%d-%b-%Y_(%H_%M_%S.%f)")

    # create data directory
    if not os.path.exists(directory):
        os.makedirs(directory)

    fig = plt.figure(figsize=(10, 7))
    plt.boxplot(timing_dict.values(), showfliers=False)
    plt.xticks([1, 2, 3, 4, 5],
               ['CAM', 'PRE', 'MASK', 'FACE', 'POST'])
    plt.ylabel('Time (s)')
    plt.xlabel('Subsystem Name')
    plt.suptitle('Subsystem Timing Performance')
    # Save the figure and show
    plt.savefig(directory + timestamp + "_box_plot.png")

    # save timing dict to csv, w/ timestamp
    file_name = timestamp + "_data.csv"
    with open(directory + file_name, "a") as file:
        for subsystem in timing_dict.keys():
            # write subsystem name
            file.write(subsystem + "\n")
            # write data
            np.savetxt(fname=file, X=timing_dict[subsystem], delimiter=",")
