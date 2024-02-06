import os
import matplotlib.pyplot as plt
import numpy as np

# Script to plot the reward trend from the reward files relatively to each trained agent

# List of all the agents 
agents = ["alpha_0", "alpha_1", "alpha_2", "alpha_3", "hivemind_0", "beta_0", "beta_1", "beta_2", "beta_3", "beta_4"]

for a in range(len(agents)):

    # Path of the reward file to read 
    reward_path = "./"+agents[a]+"/reward"

    # Final path to save the plots
    plot_path = "./"+agents[a]+"/reward_plots"

    # If the directory for the agent does not exist, continue
    if not os.path.exists("./"+agents[a]): continue

    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    # List the content of the path
    files = os.listdir(reward_path)

    # Only files with .txt extension
    txt_files = [file for file in files if file.endswith('.txt')]

    # Iterate over the files 
    for txt_file in txt_files:
        file_path = os.path.join(reward_path, txt_file)

        # Read data from file
        with open(file_path, 'r') as file:

            line = file.readline().strip()

            if line:

                values = line.split(';')

                # Numpy array for the data to plot
                data = np.array(values[1:-1], dtype=float)

                # Create a new figure
                plt.figure()

                # Plot the data
                plt.plot(data)

                # Plot title
                plt.title(f'{txt_file}')
                # Axis labels
                plt.xlabel('Generation')
                plt.ylabel('Reward')

                # Numpy array with a sequence of value between min(data) and max(data) spaced by step
                y_ticks = np.arange(min(data), max(data)+1, step=20)
                
                # Setting tick marks on the y-axis 
                plt.yticks(y_ticks)

                # Save plot                                 # removing extension ".txt" 
                image_name = os.path.join(plot_path, f'plot_{os.path.splitext(txt_file)[0]}.png')
                plt.savefig(image_name)

                # Close plot
                plt.close()
            else:
                print(f"File {txt_file} contains an empty line")



