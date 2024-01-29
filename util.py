import numpy as np
import matplotlib.pyplot as plt
import os


def plot_learning_curve(x, scores, epsilons, filename, agent_name, player_type):
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Games number", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-100):(t+1)])       #avg over 100 games if n_games > 100 else avg over n_games
    ax2.scatter(x,running_avg, color = "C1", s=20)
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Avg '+str(player_type)+' Reward', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")

    if not os.path.exists("./"+agent_name+"/train_plots"):
        os.makedirs("./"+agent_name+"/train_plots")

    plt.savefig("./"+agent_name+"/train_plots/"+filename)


def write_config(agent_name, player_name, trainer, lr, batch_size, max_memory, eps, eps_dec, eps_min, layers, reward_criterion):
    file_name = agent_name+'_'+player_name+'_config.txt'
    path = './'+agent_name+'/config/'+file_name
    if not os.path.exists("./"+agent_name+'/config'):
        os.makedirs("./"+agent_name+'/config')
    with open(path, 'w') as f:
        f.write("Agent name: "+agent_name+"\n")
        f.write("Player name: "+player_name+"\n")
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

 