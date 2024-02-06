import numpy as np
import matplotlib.pyplot as plt
import os

# A function to plot the learning curve of an agent
def plot_learning_curve(x, scores, epsilons, filename, agent_name, player_type, play=False):

    # Create a new figure
    fig = plt.figure()

    # Adding subplots to the figure
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)

    # Plot epsilon values against the x values on the first subplot
    ax.plot(x, epsilons, color="C0")

    # Set x-axis label as "Games number"
    ax.set_xlabel("Games number", color="C0")

    # Set y-axis label as "Epsilon"
    ax.set_ylabel("Epsilon", color="C0")

    # Axis color
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    # Running average of reward scores over a window of 100 games 
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-100):(t+1)])       # avg over 100 games if n_games > 100 else avg over n_games
    # Plot of the running average
    ax2.scatter(x,running_avg, color = "C1", s=20)

    # Hide the x-axis for the second subplot
    ax2.axes.get_xaxis().set_visible(False)

    # Move the y-axis ticks to the right side
    ax2.yaxis.tick_right()

    # Set y-axis label (of second subplot) as "Avg player type Reward"
    ax2.set_ylabel('Avg '+str(player_type)+' Reward', color="C1")
    # Set the position of the y-axis label to the right
    ax2.yaxis.set_label_position('right')
    # Set axis color
    ax2.tick_params(axis='y', colors="C1")

    # Training plots
    if not play :
        if not os.path.exists("./"+agent_name+"/train_plots"):
            os.makedirs("./"+agent_name+"/train_plots")

        plt.savefig("./"+agent_name+"/train_plots/"+filename)

    # Play plots
    else:
        if not os.path.exists("./"+agent_name+"/play_plots"):
            os.makedirs("./"+agent_name+"/play_plots")

        plt.savefig("./"+agent_name+"/play_plots/"+filename)

    # close the plot to avoid overhead
    plt.close()


# Funciton to print the training configuration of an agent to a txt file
def write_config(agent_name, player_name, map_name, trainer, lr, batch_size, max_memory, eps, eps_dec, eps_min, layers, reward_criterion):
    
    # File name
    file_name = agent_name+'_'+player_name+'_config.txt'

    # Path for saving the config file
    path = './'+agent_name+'/config/'+file_name

    # Check if the config directory exists
    if not os.path.exists("./"+agent_name+'/config'):
        os.makedirs("./"+agent_name+'/config')

    # Write configuration informations to the file
    with open(path, 'w') as f:
        f.write("Agent name: "+agent_name+"\n")
        f.write("Player name: "+player_name+"\n")
        f.write("Map name: "+map_name+"\n")
        f.write("Reward criterion: "+reward_criterion+"\n")
        f.write("Trainer: "+trainer+"\n")
        f.write("Learning rate: "+str(lr)+"\n")
        f.write("Batch size: "+str(batch_size)+"\n")
        f.write("Max memory size: "+str(max_memory)+"\n")
        f.write("Epsilon: "+str(eps)+"\n")
        f.write("Epsilon decrement: "+str(eps_dec)+"\n")
        f.write("Minimum epsilon: "+str(eps_min)+"\n")
        f.write("Network layers: ")
        for layer in layers:
            f.write(str(layer)+" ")

 