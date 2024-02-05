import torch
import random
from collections import deque
import numpy as np
from models import QTrainer, QTrainer_beta_1, QNet, ConvQNet
import os
import ast # to easily load memory from file (used in Hive_mind)

# For probing purposes
import time

# Default parameters for all models
MAX_MEMORY = 1000000
BATCH_SIZE = 10000
LR = 0.0005

# Agent Alpha includes the first tests with agents 0, 1, 2 and 3:
    # Agent Alpha 0 is the first complete prototype, every piece work but learning seems capped by input encoding or other unknown factors
    # Agent Alpha 1 is the second complete prototype, the aim is to improve the input encoding narrowing the possible existing problems with alpha 0
    # Agent Alpha 2 comes to life after considerations made on hivemind prototypes, what about mimicking other examples of the same game? the state space is completely changed again in advantage of a coordinate/distance representation
    # Agent Alpha 3 is a test on the alpha 1 encoding, the state is the same but removing the position and the direction to obtain a simpler representation
class Agent_alpha:
    def __init__(self, alpha=0, name='model', Qtrainer=QTrainer, lr = LR, batch_size = BATCH_SIZE, max_memory = MAX_MEMORY):
        # Agent name corresponds to the alpha generation and is used to save and load the model, configs and memory
        self.agent_name = "alpha_" + str(alpha)
        # Alpha generation number (0,1,2,3)
        self.alpha = alpha
        # Seeker or Hider
        self.name = name
        # Q_trainer class is instantiated without parameters to include it in the config file
        self.Qtrainer = Qtrainer

        # Agent hyperparameters
        self.lr = lr
        self.batch_size = batch_size
        self.max_memory = max_memory
        self.n_games = 0        # number of games played
        self.epsilon = 200      # randomness
        self.gamma = 0.9        # future expected reward discount rate

        # Agent long term and replay memory
        self.memory = deque(maxlen=MAX_MEMORY)      # agent memory, queue with maxlen to automatically pop left
        self.init_memory()                          # reload all previous memeories up to MAX_MEMORY from the memory file
        self.replay_memory = []                     # a memory buffer to selectively recollect previous experiences

        # Neural network and trainer instantiation
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # alpha_generation picks the respective network architecture for the agent
        alpha_generation = {0: [79, 256, 512, 256, 5], 1: [19, 128, 128, 6], 2: [21, 256, 36, 6], 3: [75, 128, 256, 256, 128, 6]}
        self.brain = QNet(alpha_generation[self.alpha], self.agent_name, self.name).to(self.device)
        self.trainer = self.Qtrainer(self.brain, self.lr, self.gamma)

        # Load the model if it exists
        if self.brain.load():
            print("Model loaded")

    # Load the memory from the file if it exists
    def init_memory(self):
        # Check if memory file exists
        if not os.path.exists("./alpha_"+str(self.alpha)+"/memory/" + self.name +".txt"):
            # If directory already exists, return
            if os.path.exists("./alpha_"+str(self.alpha)+"/memory"):
                return
            # Otherwise create the directory
            else:
                os.makedirs("./alpha_"+str(self.alpha)+"/memory")
                return
        # Recall last MAX_MEMORY lines from the memory file
        with open("./alpha_"+str(self.alpha)+"/memory/" + self.name +".txt", "r") as f:
            lines = f.readlines()
            if len(lines) > MAX_MEMORY:
                lines = lines[-MAX_MEMORY:]
            for line in lines:
                state, action, reward, next_state, gameover = line.split(";")
                state = np.array(state[1:-1].split(","), dtype=np.float32)
                action = np.array(action[1:-1].split(","), dtype=np.int32)
                reward = np.array(reward, dtype=np.float32)
                next_state = np.array(next_state[1:-1].split(","), dtype=np.float32)
                gameover = np.array(gameover == 'True')
                self.memory.append((state, action, reward, next_state, gameover))

    # Load the replay memory SELECTIVELY from the file (when  the function is called the file already exists)
    def load_replay_memory(self, criterion="reward"):
        with open("./alpha_"+str(self.alpha)+"/memory/" + self.name +".txt", "r") as f:
            # Define different sorting criteria for existing memory - note: in memory "reward" is the 3rd element of the lines
            if criterion == "abs_reward":
                crit = lambda x: abs(float(x.split(";")[2]))    # highest reward's absolute value
                reverse = True
            elif criterion == "reward":
                crit = lambda x: float(x.split(";")[2])         # highest rewards
                reverse = True
            elif criterion == "neg_reward":
                crit = lambda x: float(x.split(";")[2])         # lowest rewards
                reverse = False
            elif criterion == "lowest_abs_reward":
                crit = lambda x: abs(float(x.split(";")[2]))    # lowest reward's absolute value
                reverse = False

            # Note: the lines are picked from the bottom of the file -> reverse = True picks the highest values

            lines = sorted(f.readlines(), key=crit, reverse=reverse)
        
        # Load up to batch_size lines
        if len(lines) > self.batch_size:
            lines = lines[:self.batch_size]
        for line in lines:
            state, action, reward, next_state, gameover = line.split(";")
            state = np.array(state[1:-1].split(","), dtype=np.float32)
            action = np.array(action[1:-1].split(","), dtype=np.int32)
            reward = np.array(float(reward), dtype=np.float32)
            next_state = np.array(next_state[1:-1].split(","), dtype=np.float32)
            gameover = np.array(gameover == 'True')
            self.replay_memory.append((state, action, reward, next_state, gameover))

    # Get the state of the game for the agent - Using different encodings/data for different alpha generations       
    def get_state(self, game, player):
        x = player.x
        y = player.y

        i = y // player.size
        j = x // player.size

        # Alpha 0
        if self.alpha == 0:
            # Normalized position of the player
            x_norm = player.x / (game.width - player.size)
            y_norm = player.y / (game.height - player.size)

            # What the player sees according to policy
            view = player.view

            # One-hot encoding of all the object types - None is either outside the map or out of the player's view (hidden by an object) - Player.mask_view()
            objects = {'wall': '10000', 'floor': '01000', 'hider': '00100', 'movable_wall': '00010','seeker': '00001', None: '00000'}

            # Encode the view
            view_vector = []
            for l in view:
                for c in l:
                    if c is None:
                        for n in objects[c]:
                            view_vector.append(int(n))
                    else:
                        for n in objects[c.obj_type]:
                            view_vector.append(int(n))

            # Depending on player's direction the neighbourhood is composed of the 3 cells around the player (except for the faced one)
            neighbourhood = []      # Order - left back right

            # None is outside of the map
            if player.direction == 'u':     # up
                left = player.map[i][j-1].obj_type if j-1 >= 0 else None
                back = player.map[i+1][j].obj_type if i+1 < game.rows else None
                right = player.map[i][j+1].obj_type if j+1 < game.cols else None

            elif player.direction == 'd':       # down
                left = player.map[i][j+1].obj_type if j+1 < game.cols else None
                back = player.map[i-1][j].obj_type if i-1 >= 0 else None
                right = player.map[i][j-1].obj_type if j-1 >= 0 else None

            elif player.direction == 'l':       # left
                left = player.map[i+1][j].obj_type if i+1 < game.rows else None
                back = player.map[i][j+1].obj_type if j+1 < game.cols else None
                right = player.map[i-1][j].obj_type if i-1 >= 0 else None

            elif player.direction == 'r':       # right
                left = player.map[i-1][j].obj_type if i-1 >= 0 else None
                back = player.map[i][j-1].obj_type if j-1 >= 0 else None
                right = player.map[i+1][j].obj_type if i+1 < game.rows else None
            
            # Encode the neighbourhood
            for n in objects[left]:
                neighbourhood.append(int(n))
            for n in objects[back]:
                neighbourhood.append(int(n))
            for n in objects[right]:
                neighbourhood.append(int(n))

            # Other player's position and normalization
            other_player = game.players[0] if player.obj_type == 'seeker' else game.players[1]
            other_player_x = other_player.x / (game.width - other_player.size)
            other_player_y = other_player.y / (game.height - other_player.size)


            state = [x_norm,y_norm] + view_vector + neighbourhood + [other_player_x, other_player_y]
        # Alpha 1
        elif self.alpha == 1:
            # What the player sees according to policy
            view = player.view

            # Encoding of all the object types - None is either outside the map or out of the player's view (hidden by an object) - Player.mask_view()
            objects = {'wall': '5', 'floor': '1', 'hider': '100', 'movable_wall': '10','seeker': '100', None: '0'}
            
            # Encode the view
            view_vector = []
            for l in view:
                for c in l:
                    if c is None:
                        view_vector.append(int(objects[c]))
                    else:
                        view_vector.append(int(objects[c.obj_type]))
                        
            # Depending on player's direction the neighbourhood is composed of the 3 cells around the player (except for the faced one)
            neighbourhood = [] # left back right

            # None is outside of the map
            if player.direction == 'u':         # up
                left = player.map[i][j-1].obj_type if j-1 >= 0 else None
                back = player.map[i+1][j].obj_type if i+1 < game.rows else None
                right = player.map[i][j+1].obj_type if j+1 < game.cols else None

            elif player.direction == 'd':       # down
                left = player.map[i][j+1].obj_type if j+1 < game.cols else None
                back = player.map[i-1][j].obj_type if i-1 >= 0 else None
                right = player.map[i][j-1].obj_type if j-1 >= 0 else None

            elif player.direction == 'l':       # left
                left = player.map[i+1][j].obj_type if i+1 < game.rows else None
                back = player.map[i][j+1].obj_type if j+1 < game.cols else None
                right = player.map[i-1][j].obj_type if i-1 >= 0 else None

            elif player.direction == 'r':       # right
                left = player.map[i-1][j].obj_type if i-1 >= 0 else None
                back = player.map[i][j-1].obj_type if j-1 >= 0 else None
                right = player.map[i+1][j].obj_type if i+1 < game.rows else None
            
            # Encode the neighbourhood
            neighbourhood.append(int(objects[left]))
            neighbourhood.append(int(objects[back]))
            neighbourhood.append(int(objects[right]))

            # Other player's position and normalization
            other_player = game.players[0] if player.obj_type == 'seeker' else game.players[1]
            other_player_i = other_player.y // other_player.size
            other_player_j = other_player.x // other_player.size


            state = [i,j] + view_vector + neighbourhood + [other_player_i, other_player_j]
        # Alpha 2
        elif self.alpha == 2:

            # Other player's position and normalization
            other_player = game.players[0] if player.obj_type == 'seeker' else game.players[1]
            other_player_i = other_player.y // other_player.size
            other_player_j = other_player.x // other_player.size

            # Euclidean distance from the other player
            distance = np.sqrt((other_player_i - i)**2 + (other_player_j - j)**2)
            # Normalization of the distance
            distance = distance / (game.rows * np.sqrt(2))    # Assuming a square map

            # Direction of the player
            if player.direction == 'u':
                direction = 0
            elif player.direction == 'd':
                direction = 1
            elif player.direction == 'l':
                direction = 2
            elif player.direction == 'r':
                direction = 3

            # Encoding of all the object types - None is either outside the map or out of the player's view (hidden by an object) - Player.mask_view()
            objects = {'wall': '5', 'floor': '1', 'hider': '100', 'movable_wall': '10','seeker': '100', None: '0'}

            # What the player sees according to policy
            view = player.view

            # Encode the view
            view_vector = []
            for l in view:
                for c in l:
                    if c is None:
                        view_vector.append(int(objects[c]))
                    else:
                        view_vector.append(int(objects[c.obj_type])/100)

            # Depending on player's direction the neighbourhood is composed of the 3 cells around the player (except for the faced one)
            neighbourhood = []

            # None is outside of the map
            if player.direction == 'u':         # up
                left = player.map[i][j-1].obj_type if j-1 >= 0 else None
                back = player.map[i+1][j].obj_type if i+1 < game.rows else None
                right = player.map[i][j+1].obj_type if j+1 < game.cols else None

            elif player.direction == 'd':       # down
                left = player.map[i][j+1].obj_type if j+1 < game.cols else None
                back = player.map[i-1][j].obj_type if i-1 >= 0 else None
                right = player.map[i][j-1].obj_type if j-1 >= 0 else None

            elif player.direction == 'l':       # left
                left = player.map[i+1][j].obj_type if i+1 < game.rows else None
                back = player.map[i][j+1].obj_type if j+1 < game.cols else None
                right = player.map[i-1][j].obj_type if i-1 >= 0 else None

            elif player.direction == 'r':       # right
                left = player.map[i-1][j].obj_type if i-1 >= 0 else None
                back = player.map[i][j-1].obj_type if j-1 >= 0 else None
                right = player.map[i+1][j].obj_type if i+1 < game.rows else None

            neighbourhood.append(int(objects[left]))
            neighbourhood.append(int(objects[back]))
            neighbourhood.append(int(objects[right]))

            # Other player's position and normalization
            other_player_i = other_player_i / (game.rows - 1)
            other_player_j = other_player_j / (game.cols - 1)

            state = [i,j] + view_vector + neighbourhood + [distance, direction] + [other_player_i, other_player_j]
        # Alpha 3
        elif self.alpha == 3:

            # What the player sees according to policy
            view = player.view

            # One-hot encoding of all the object types - None is either outside the map or out of the player's view (hidden by an object) - Player.mask_view()
            objects = {'wall': '10000', 'floor': '01000', 'hider': '00100', 'movable_wall': '00010', 'seeker': '00001', None: '00000'}
            
            # Encode the view
            view_vector = []
            for l in view:
                for c in l:
                    if c is None:
                        for n in objects[c]:
                            view_vector.append(int(n))
                    else:
                        for n in objects[c.obj_type]:
                            view_vector.append(int(n))
            
            # Depending on player's direction the neighbourhood is composed of the 3 cells around the player (except for the faced one)
            neighbourhood = []

            # None is outside of the map
            if player.direction == 'u':         # up
                left = player.map[i][j - 1].obj_type if j - 1 >= 0 else None
                back = player.map[i + 1][j].obj_type if i + 1 < game.rows else None
                right = player.map[i][j + 1].obj_type if j + 1 < game.cols else None

            elif player.direction == 'd':       # down
                left = player.map[i][j + 1].obj_type if j + 1 < game.cols else None
                back = player.map[i - 1][j].obj_type if i - 1 >= 0 else None
                right = player.map[i][j - 1].obj_type if j - 1 >= 0 else None

            elif player.direction == 'l':       # left
                left = player.map[i + 1][j].obj_type if i + 1 < game.rows else None
                back = player.map[i][j + 1].obj_type if j + 1 < game.cols else None
                right = player.map[i - 1][j].obj_type if i - 1 >= 0 else None

            elif player.direction == 'r':       # right
                left = player.map[i - 1][j].obj_type if i - 1 >= 0 else None
                back = player.map[i][j - 1].obj_type if j - 1 >= 0 else None
                right = player.map[i + 1][j].obj_type if i + 1 < game.rows else None

            # Encode the neighbourhood
            for n in objects[left]:
                neighbourhood.append(int(n))
            for n in objects[back]:
                neighbourhood.append(int(n))
            for n in objects[right]:
                neighbourhood.append(int(n))

            state = view_vector + neighbourhood

        return state
    
    # Pick the next action to take - Tradeoff exploration / exploitation
    def get_action(self, state):
        # Final action is a one-hot encoded vector
        final_action = [0 for i in range(self.brain.layer_list[-1])]        # Not every Alpha agent employ the 6th action (standing still)

        # Move randomly
        if random.randint(0, 200) < self.epsilon:   # 200 is arbitrary
            action = random.randint(0, (self.brain.layer_list[-1]-1))
            final_action[action] = 1

        # Move according to the policy network
        else:
            state = np.array(state)
            current_state = torch.tensor(state, dtype=torch.float).to(self.device)
            prediction = self.brain(current_state)
            action = torch.argmax(prediction).item()
            final_action[action] = 1

        return final_action

    # Store the experience in the agent's memory and in the memory file
    def remember(self, state, action, reward, next_state, gameover):
        self.memory.append((state, action, reward, next_state, gameover))
        
        with open("./alpha_"+str(self.alpha)+"/memory/" + self.name +".txt", "a") as f:
            f.write(str(state) + ";")
            f.write(str(action) + ";")
            f.write(str(reward) + ";")
            f.write(str(next_state) + ";")
            f.write(str(gameover) + "\n")

    # Online training - Short term memory
    def train_short_memory(self, state, action, reward, next_state, gameover):
        # Cast to numpy arrays for the torch.tensor conversion
        state = np.array(state)
        next_state = np.array(next_state)
        action = np.array(action)
        self.trainer.train_step(state, action, reward, next_state, gameover)

    # Batch training - Long term memory
    def train_long_memory(self):
        start = time.time()
        # Decrease exploration probability
        if self.epsilon > 0 : self.epsilon -= 1 

        # Sample a random batch from the memory
        if len(self.memory) > self.batch_size:
            batch_sample = random.sample(self.memory, self.batch_size)

        # If the memory is not big enough, train on the whole memory
        else:
            batch_sample = self.memory

        # Separate the batch into its components
        states, actions, rewards, next_states, gameovers = zip(*batch_sample)

        # Convert to numpy arrays for the torch.tensor conversion
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        gameovers = np.array(gameovers)

        self.trainer.train_step(states, actions, rewards, next_states, gameovers)

        # Save the model
        self.brain.save()

        # Log time took for training
        end = time.time()
        if self.name == "seeker": print(f"\033[94mtraning seeker's long memory took: {end - start} seconds\033[0m")
        else: print(f"\033[92mtraning hider's long memory took: {end - start} seconds\033[0m")

    # Replay training - Selective memory
    def train_replay(self, criterion="reward"):
        start = time.time()

        # Load the replay memory according to the criterion
        self.load_replay_memory(criterion)

        # Separate the replay memory into its components
        states, actions, rewards, next_states, gameovers = zip(*self.replay_memory)

        # Convert to numpy arrays for the torch.tensor conversion
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        gameovers = np.array(gameovers)

        self.trainer.train_step(states, actions, rewards, next_states, gameovers)

        # Log time took for training
        end = time.time()
        print("+"*50)
        print(f"\033[96mtrain_replay with {criterion} criterion in: {end - start} seconds\033[0m")
        print("+"*50)

        # Reset the replay memory for next training
        self.replay_memory = []
    
    # Clean the memory file from perfectly identical consecutive lines if their number is over a certain threshold
    def clean_memory(self, duplicates=100):
        start = time.time()

        file_path = "./alpha_"+str(self.alpha)+"/memory/" + self.name + ".txt"
        with open(file_path, "r") as f:
            lines = f.readlines()
        
        count = 0
        erased = 0

        i = 0
        while i < len(lines) - 1:
            if lines[i] == lines[i + 1]:
                count += 1
                if count == duplicates:
                                    # First duplicate     # Next value
                    lines = lines[:i - duplicates + 2] + lines[i + 1:]
                    count = 0
                    erased += duplicates
            else:
                count = 0
            i += 1

        # Update the memory file
        with open(file_path, "w") as f:
            f.writelines(lines)

        # Log time took for cleaning
        end = time.time()
        print("#" * 50)
        print(f"\033[92mclean_memory for {self.name}, erased {erased} lines in: ", end - start, " seconds\033[0m")
        print("#" * 50)

# Hivemind, as the name suggests, is meant to be able to predict next action taking into account the whole map image - It's the only Agent that takes advantage of the Convolutional Neural Network
class Agent_hivemind:
    def __init__(self, name='model', Qtrainer=QTrainer, lr = LR, batch_size = BATCH_SIZE, max_memory = MAX_MEMORY):
        # Agent name is used to save and load the model, configs and memory
        self.agent_name = "hivemind"
        # Seeker or Hider
        self.name = name
        # Q_trainer class is instantiated without parameters to include it in the config file
        self.Qtrainer = Qtrainer

        # Agent hyperparameters
        self.lr = lr
        self.batch_size = batch_size
        self.max_memory = max_memory
        self.n_games = 0    # number of games played
        self.epsilon = 200  # randomness
        self.gamma = 0.9    # future expected reward discount rate

        # Agent long term and replay memory
        self.memory = deque(maxlen=self.max_memory)         # agent memory, queue with maxlen to automatically pop left
        self.init_memory()                                  # reload all previous memeories up to MAX_MEMORY from the memory file
        self.replay_memory = []                             # a memory buffer to selectively recollect previous experiences

        # Neural network and trainer instantiation
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # brain designed for a 26x26 map
        self.brain = ConvQNet([[1, 3, 3, 1, 1], [3, 1, 9, 1, 0], [1, 1, 7, 1, 0], [1, 1, 5, 1, 0]], [64, 128, 128, 6], self.agent_name, self.name).to(self.device)
        # Convolutional layers definition is not flexible and needs a coherent input (i.e. Map size dependent)
        self.trainer = self.Qtrainer(self.brain, self.lr, self.gamma, convolutional=True)

        if self.brain.load():
            print("Model loaded")

        print(f"AGENT HIVEMIND: training {self.name} with {self.device} device")
        
    def init_memory(self):
        # check if memory file exists
        if not os.path.exists("./hivemind_0/memory/" + self.name +".txt"):
            if os.path.exists("./hivemind_0/memory"):
                return
            else:
                os.makedirs("./hivemind_0/memory")
                return
        # recall last lines of memory up to MAX_MEMORY
        with open("./hivemind_0/memory/" + self.name +".txt", "r") as f:
            lines = f.readlines()
            if len(lines) > self.max_memory:
                lines = lines[-self.max_memory:]
            for line in lines:
                state, action, reward, next_state, gameover = line.split(";")
                state = np.array(ast.literal_eval(state), dtype=np.float32)
                action = np.array(action[1:-1].split(","), dtype=np.float32)
                reward = np.array(float(reward), dtype=np.float32)
                next_state = np.array(ast.literal_eval(next_state), dtype=np.float32)
                gameover = np.array(gameover == 'True')
                self.memory.append((state, action, reward, next_state, gameover))

    def load_replay_memory(self, criterion="reward"):
        with open("./hivemind_0/memory/" + self.name +".txt", "r") as f:
            if criterion == "abs_reward":
                crit = lambda x: abs(float(x.split(";")[2]))
                reverse = True
            elif criterion == "reward":
                crit = lambda x: float(x.split(";")[2])
                reverse = True
            elif criterion == "neg_reward":
                crit = lambda x: float(x.split(";")[2])
                reverse = False
            elif criterion == "lowest_abs_reward":
                crit = lambda x: abs(float(x.split(";")[2]))
                reverse = False

            lines = sorted(f.readlines(), key=crit, reverse=reverse)

        if len(lines) > self.batch_size:
            lines = lines[:self.batch_size]
        for line in lines:
            state, action, reward, next_state, gameover = line.split(";")
            state = np.array(ast.literal_eval(state), dtype=np.float32)
            action = np.array(action[1:-1].split(","), dtype=np.float32)
            reward = np.array(float(reward), dtype=np.float32)
            next_state = np.array(ast.literal_eval(next_state), dtype=np.float32)
            gameover = np.array(gameover == 'True')
            self.replay_memory.append((state, action, reward, next_state, gameover))

    # hivemind agent is designed to acquire the matrix of the whole map
    def get_state(self, game, player):
        start = time.time()
        objects = {'wall': '5', 'floor': '1', 'hider': '100', 'movable_wall': '10','seeker': '-100', None: '0'}
        # ["wall", "floor", "hider", "movable_wall", None]
        state = []
        for row in range(len(player.map)):
            state.append([])
            for cell in range(len(player.map[row])):
                state[row].append(int(objects[game.map[row][cell].obj_type]))

        return state

    def get_action(self, state):
        start = time.time()
        # tradeoff exploration / exploitation
        self.epsilon = self.randomness - self.n_games
        # Final action is a one-hot encoded vector
        final_action = [0,0,0,0,0,0]
        if random.randint(0, 200) < self.epsilon:   # 200 is arbitrary
            action = random.randint(0, 5)
            final_action[action] = 1
        else:
            state = np.array(state)
            current_state = torch.tensor(state, dtype=torch.float).to(self.device)
            prediction = self.brain(current_state)
            action = torch.argmax(prediction).item()
            final_action[action] = 1
        
        end = time.time()
        #print("get_action: ", end - start)

        return final_action

    def remember(self, state, action, reward, next_state, gameover):
        self.memory.append((state, action, reward, next_state, gameover))
        # append to a certain file
        
        with open("./hivemind_0/memory/" + self.name +".txt", "a") as f:
            f.write(str(state) + ";")
            f.write(str(action) + ";")
            f.write(str(reward) + ";")
            f.write(str(next_state) + ";")
            f.write(str(gameover) + "\n")
        
    def train_short_memory(self, state, action, reward, next_state, gameover):
        start = time.time()
        state = np.array(state)
        next_state = np.array(next_state)
        self.trainer.train_step(state, action, reward, next_state, gameover)
        end = time.time()
        #print("train_short_memory: ", end - start)

    def train_long_memory(self):
        start = time.time()
        if len(self.memory) > self.batch_size:
            batch_sample = random.sample(self.memory, self.batch_size)
        else:
            batch_sample = self.memory

        states, actions, rewards, next_states, gameovers = zip(*batch_sample)

        # convert to numpy arrays
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        gameovers = np.array(gameovers)

        self.trainer.train_step(states, actions, rewards, next_states, gameovers)

        self.brain.save()
        end = time.time()
        if self.name == "seeker": print(f"\033[94mtraning seeker's long memory took: {end - start} seconds\033[0m")
        else: print(f"\033[92mtraning hider's long memory took: {end - start} seconds\033[0m")

    def train_replay(self, criterion="reward"):
        start = time.time()

        self.load_replay_memory(criterion)

        states, actions, rewards, next_states, gameovers = zip(*self.replay_memory)

        # convert to numpy arrays
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        gameovers = np.array(gameovers)

        self.trainer.train_step(states, actions, rewards, next_states, gameovers)

        end = time.time()
        print("+"*50)
        print(f"\033[96mtrain_replay with {criterion} criterion in: {end - start} seconds\033[0m")
        print("+"*50)
        self.replay_memory = []

    def clean_memory(self, duplicates=100):
        start = time.time()
        # clean identical lines if number is over
        file_path = "./hivemind_0/memory/" + self.name + ".txt"
        with open(file_path, "r") as f:
            lines = f.readlines()
        count = 0
        erased = 0
        i = 0
        while i < len(lines) - 1:
            if lines[i] == lines[i + 1]:
                count += 1
                if count == duplicates:
                    lines = lines[:i - duplicates + 2] + lines[i + 1:]
                    count = 0
                    erased += duplicates
            else:
                count = 0
            i += 1

        with open(file_path, "w") as f:
            f.writelines(lines)

        end = time.time()
        print("#" * 50)
        print(f"\033[92mclean_memory for {self.name}, erased {erased} lines in: ", end - start, " seconds\033[0m")
        print("#" * 50)

class Agent_alpha_4:
    def __init__(self, name='model', Qtrainer=QTrainer_beta_1, lr=0.0005, batch_size=1000, max_memory=100000, eps_dec= 5e-4, eps_min = 0.01):
        self.agent_name = "alpha_4"
        self.name = name
        self.Qtrainer = Qtrainer
        self.lr = lr
        self.batch_size = batch_size
        self.max_memory = max_memory
        self.n_games = 0
        self.epsilon = 1.0
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=self.max_memory)  # automatic popleft()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.brain = QNet([75, 256, 256, 6], self.agent_name, self.name).to(self.device)
        self.trainer = self.Qtrainer(self.brain, self.lr, self.gamma)

        if self.brain.load():
            print("Model loaded")

        print(f"AGENT ALPHA 4: training {self.name} with {self.device} device")

    def get_state(self, game, player):
        x = player.x
        y = player.y
        i = y // player.size
        j = x // player.size

        view = player.view
        objects = {'wall': '10000', 'floor': '01000', 'hider': '00100', 'movable_wall': '00010', 'seeker': '00001', None: '00000'}
        # ["wall", "floor", "hider", "movable_wall", None]
        view_vector = []
        for l in view:
            for c in l:
                if c is None:
                    for n in objects[c]:
                        view_vector.append(int(n))
                else:
                    for n in objects[c.obj_type]:
                        view_vector.append(int(n))

        neighbourhood = []  # left back right

        if player.direction == 'u':
            left = player.map[i][j - 1].obj_type if j - 1 >= 0 else None
            back = player.map[i + 1][j].obj_type if i + 1 < game.rows else None
            right = player.map[i][j + 1].obj_type if j + 1 < game.cols else None

        elif player.direction == 'd':
            left = player.map[i][j + 1].obj_type if j + 1 < game.cols else None
            back = player.map[i - 1][j].obj_type if i - 1 >= 0 else None
            right = player.map[i][j - 1].obj_type if j - 1 >= 0 else None

        elif player.direction == 'l':
            left = player.map[i + 1][j].obj_type if i + 1 < game.rows else None
            back = player.map[i][j + 1].obj_type if j + 1 < game.cols else None
            right = player.map[i - 1][j].obj_type if i - 1 >= 0 else None

        elif player.direction == 'r':
            left = player.map[i - 1][j].obj_type if i - 1 >= 0 else None
            back = player.map[i][j - 1].obj_type if j - 1 >= 0 else None
            right = player.map[i + 1][j].obj_type if i + 1 < game.rows else None

        for n in objects[left]:
            neighbourhood.append(int(n))
        for n in objects[back]:
            neighbourhood.append(int(n))
        for n in objects[right]:
            neighbourhood.append(int(n))

        state = view_vector + neighbourhood
        return state

    def get_action(self, state):
        # tradeoff exploration / exploitation
        final_action = [0, 0, 0, 0, 0, 0]
        if np.random.random() > self.epsilon:
            state = np.array(state)
            current_state = torch.tensor(state, dtype=torch.float).to(self.device)
            prediction = self.brain(current_state)
            action = torch.argmax(prediction).item()
            final_action[action] = 1
        else:
            action = random.randint(0, 5)
            final_action[action] = 1

        return final_action

    def remember(self, state, action, reward, next_state, gameover):
        self.memory.append((state, action, reward, next_state, gameover))

    def train_short_memory(self, state, action, reward, next_state, gameover):
        start = time.time()
        state = np.array(state)
        next_state = np.array(next_state)
        self.trainer.train_step(state, action, reward, next_state, gameover)
        end = time.time()

    def train_long_memory(self):
        if len(self.memory) < self.batch_size:
            return
        else:
            batch_sample = random.sample(self.memory, self.batch_size)

        states, actions, rewards, next_states, gameovers = zip(*batch_sample)

        # convert to numpy arrays
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        gameovers = np.array(gameovers)

        self.trainer.train_step(states, actions, rewards, next_states, gameovers)

        self.brain.save()

        self.decrement_epsilon()

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

class Agent_alpha_5:
    def __init__(self, name='model', Qtrainer=QTrainer_beta_1, lr=0.001, batch_size=1000, max_memory=100000, epsilon = 1.0, eps_dec= 5e-4, eps_min = 0.05):
        self.agent_name = "alpha_5"
        self.name = name
        self.Qtrainer = Qtrainer
        self.lr = lr
        self.batch_size = batch_size
        self.max_memory = max_memory
        self.n_games = 0
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=self.max_memory)  # automatic popleft()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.brain = QNet([81, 256, 256, 6], self.agent_name, self.name).to(self.device)
        self.trainer = self.Qtrainer(self.brain, self.lr, self.gamma)

        if self.brain.load():
            print("Model loaded")

        print(f"AGENT ALPHA 5: training {self.name} with {self.device} device")

    def get_state(self, game, player):
        start = time.time()
        x = player.x
        y = player.y
        i = y // player.size
        j = x // player.size

        other_player = game.players[0] if player.obj_type == 'seeker' else game.players[1]
        other_player_i = other_player.y // other_player.size
        other_player_j = other_player.x // other_player.size
        distance = np.sqrt((other_player_i - i)**2 + (other_player_j - j)**2)
        direction = player.direction

        if direction == 'u':
            direction = 0
        elif direction == 'd':
            direction = 1
        elif direction == 'l':
            direction = 2
        elif direction == 'r':
            direction = 3

        view = player.view
        objects = {'wall': '10000', 'floor': '01000', 'hider': '00100', 'movable_wall': '00010', 'seeker': '00001', None: '00000'}
        # ["wall", "floor", "hider", "movable_wall", None]
        view_vector = []
        for l in view:
            for c in l:
                if c is None:
                    for n in objects[c]:
                        view_vector.append(int(n))
                else:
                    for n in objects[c.obj_type]:
                        view_vector.append(int(n))

        neighbourhood = []  # left back right

        if player.direction == 'u':
            left = player.map[i][j - 1].obj_type if j - 1 >= 0 else None
            back = player.map[i + 1][j].obj_type if i + 1 < game.rows else None
            right = player.map[i][j + 1].obj_type if j + 1 < game.cols else None

        elif player.direction == 'd':
            left = player.map[i][j + 1].obj_type if j + 1 < game.cols else None
            back = player.map[i - 1][j].obj_type if i - 1 >= 0 else None
            right = player.map[i][j - 1].obj_type if j - 1 >= 0 else None

        elif player.direction == 'l':
            left = player.map[i + 1][j].obj_type if i + 1 < game.rows else None
            back = player.map[i][j + 1].obj_type if j + 1 < game.cols else None
            right = player.map[i - 1][j].obj_type if i - 1 >= 0 else None

        elif player.direction == 'r':
            left = player.map[i - 1][j].obj_type if i - 1 >= 0 else None
            back = player.map[i][j - 1].obj_type if j - 1 >= 0 else None
            right = player.map[i + 1][j].obj_type if i + 1 < game.rows else None

        for n in objects[left]:
            neighbourhood.append(int(n))
        for n in objects[back]:
            neighbourhood.append(int(n))
        for n in objects[right]:
            neighbourhood.append(int(n))

        distance = distance / (game.rows + game.cols)

        state = [i,j] + view_vector + neighbourhood + [distance, direction] + [other_player_i, other_player_j]
        return state

    def get_action(self, state):
        # tradeoff exploration / exploitation
        final_action = [0, 0, 0, 0, 0, 0]
        if np.random.random() > self.epsilon:
            state = np.array(state)
            current_state = torch.tensor(state, dtype=torch.float).to(self.device)
            prediction = self.brain(current_state)
            action = torch.argmax(prediction).item()
            final_action[action] = 1
        else:
            action = random.randint(0, 5)
            final_action[action] = 1

        return final_action
    
    def perform_action(self, state):
        # only exploitation
        final_action = [0, 0, 0, 0, 0, 0]
        
        state = np.array(state)
        current_state = torch.tensor(state, dtype=torch.float).to(self.device)
        prediction = self.brain(current_state)
        action = torch.argmax(prediction).item()
        final_action[action] = 1

        return final_action

    def remember(self, state, action, reward, next_state, gameover):
        self.memory.append((state, action, reward, next_state, gameover))

    def train_short_memory(self, state, action, reward, next_state, gameover):
        start = time.time()
        state = np.array(state)
        next_state = np.array(next_state)
        self.trainer.train_step(state, action, reward, next_state, gameover)
        end = time.time()

    def train_long_memory(self):
        if len(self.memory) < self.batch_size:
            return
        else:
            batch_sample = random.sample(self.memory, self.batch_size)

        states, actions, rewards, next_states, gameovers = zip(*batch_sample)

        # convert to numpy arrays
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        gameovers = np.array(gameovers)

        self.trainer.train_step(states, actions, rewards, next_states, gameovers)

        self.brain.save()

        self.decrement_epsilon()

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

class Agent_alpha_6:
    def __init__(self, name='model', Qtrainer=QTrainer_beta_1, lr=0.001, batch_size=1000, max_memory=100000, epsilon = 1.0, eps_dec= 5e-4, eps_min = 0.05):
        self.agent_name = "alpha_6"
        self.name = name
        self.Qtrainer = Qtrainer
        self.lr = lr
        self.batch_size = batch_size
        self.max_memory = max_memory
        self.n_games = 0
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=self.max_memory)  # automatic popleft()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.brain = QNet([83, 256, 128, 128, 256, 6], self.agent_name, self.name).to(self.device)
        self.trainer = self.Qtrainer(self.brain, self.lr, self.gamma)

        if self.brain.load():
            print("Model loaded")

        print(f"AGENT ALPHA 6: training {self.name} with {self.device} device")

    def get_state(self, game, player):
        start = time.time()
        x = player.x
        y = player.y
        i = y // player.size
        j = x // player.size

        other_player = game.players[0] if player.obj_type == 'seeker' else game.players[1]
        other_player_i = other_player.y // other_player.size
        other_player_j = other_player.x // other_player.size

        directions = {'u' : '0001', 'd' : '0010', 'l' : '0100', 'r' : '1000'}

        view = player.view
        objects = {'wall': '10000', 'floor': '01000', 'hider': '00100', 'movable_wall': '00010', 'seeker': '00001', None: '00000'}
        # ["wall", "floor", "hider", "movable_wall", None]
        view_vector = []
        for l in view:
            for c in l:
                if c is None:
                    for n in objects[c]:
                        view_vector.append(int(n))
                else:
                    for n in objects[c.obj_type]:
                        view_vector.append(int(n))

        neighbourhood = []  # left back right

        if player.direction == 'u':
            left = player.map[i][j - 1].obj_type if j - 1 >= 0 else None
            back = player.map[i + 1][j].obj_type if i + 1 < game.rows else None
            right = player.map[i][j + 1].obj_type if j + 1 < game.cols else None

        elif player.direction == 'd':
            left = player.map[i][j + 1].obj_type if j + 1 < game.cols else None
            back = player.map[i - 1][j].obj_type if i - 1 >= 0 else None
            right = player.map[i][j - 1].obj_type if j - 1 >= 0 else None

        elif player.direction == 'l':
            left = player.map[i + 1][j].obj_type if i + 1 < game.rows else None
            back = player.map[i][j + 1].obj_type if j + 1 < game.cols else None
            right = player.map[i - 1][j].obj_type if i - 1 >= 0 else None

        elif player.direction == 'r':
            left = player.map[i - 1][j].obj_type if i - 1 >= 0 else None
            back = player.map[i][j - 1].obj_type if j - 1 >= 0 else None
            right = player.map[i + 1][j].obj_type if i + 1 < game.rows else None

        for n in objects[left]:
            neighbourhood.append(int(n))
        for n in objects[back]:
            neighbourhood.append(int(n))
        for n in objects[right]:
            neighbourhood.append(int(n))

        direction = []

        for n in directions[player.direction]:
            direction.append(int(n))


        state = [i,j] + view_vector + neighbourhood + direction + [other_player_i, other_player_j]

        return state

    def get_action(self, state):
        # tradeoff exploration / exploitation
        final_action = [0, 0, 0, 0, 0, 0]
        if np.random.random() > self.epsilon:
            state = np.array(state)
            current_state = torch.tensor(state, dtype=torch.float).to(self.device)
            prediction = self.brain(current_state)
            action = torch.argmax(prediction).item()
            final_action[action] = 1
        else:
            action = random.randint(0, 5)
            final_action[action] = 1

        return final_action
    
    #method to perform only model prediction in game
    def perform_action(self, state):
        # only exploitation
        final_action = [0, 0, 0, 0, 0, 0]
        
        state = np.array(state)
        current_state = torch.tensor(state, dtype=torch.float).to(self.device)
        prediction = self.brain(current_state)
        action = torch.argmax(prediction).item()
        final_action[action] = 1

        return final_action

    def remember(self, state, action, reward, next_state, gameover):
        self.memory.append((state, action, reward, next_state, gameover))

    def train_short_memory(self, state, action, reward, next_state, gameover):
        start = time.time()
        state = np.array(state)
        next_state = np.array(next_state)
        self.trainer.train_step(state, action, reward, next_state, gameover)
        end = time.time()

    def train_long_memory(self):
        if len(self.memory) < self.batch_size:
            return
        else:
            batch_sample = random.sample(self.memory, self.batch_size)

        states, actions, rewards, next_states, gameovers = zip(*batch_sample)

        # convert to numpy arrays
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        gameovers = np.array(gameovers)

        self.trainer.train_step(states, actions, rewards, next_states, gameovers)

        self.brain.save()

        self.decrement_epsilon()

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

class Agent_alpha_7:
    def __init__(self, name='model', Qtrainer=QTrainer_beta_1, lr=0.001, batch_size=1000, max_memory=100000, epsilon = 1.0, eps_dec= 5e-4, eps_min = 0.05):
        self.agent_name = "alpha_7"
        self.name = name
        self.Qtrainer = Qtrainer
        self.lr = lr
        self.batch_size = batch_size
        self.max_memory = max_memory
        self.n_games = 0
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=self.max_memory)  # automatic popleft()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.brain = QNet([28, 256, 128, 64, 32, 6], self.agent_name, self.name).to(self.device)
        self.trainer = self.Qtrainer(self.brain, self.lr, self.gamma)

        if self.brain.load():
            print("Model loaded")

        print(f"AGENT ALPHA 7: training {self.name} with {self.device} device")

    def get_state(self, game, player):
        start = time.time()
        x = player.x
        y = player.y
        i = y // player.size
        j = x // player.size

        other_player = game.players[0] if player.obj_type == 'seeker' else game.players[1]
        other_player_i = other_player.y // other_player.size
        other_player_j = other_player.x // other_player.size

        objects = {'wall': '100000', 'floor': '010000', 'hider': '001000', 'movable_wall': '000100', 'seeker': '000010', 'map_edge': '000001'}

        neighbourhood = []  

        left = player.map[i][j - 1].obj_type if j - 1 >= 0 else 'map_edge'
        back = player.map[i + 1][j].obj_type if i + 1 < game.rows else 'map_edge'
        right = player.map[i][j + 1].obj_type if j + 1 < game.cols else 'map_edge'
        up = player.map[i-1][j].obj_type if i >= 1 else 'map_edge'

        for n in objects[left]:
            neighbourhood.append(int(n))
        for n in objects[back]:
            neighbourhood.append(int(n))
        for n in objects[right]:
            neighbourhood.append(int(n))
        for n in objects[up]:
            neighbourhood.append(int(n))

        state = [i,j] + neighbourhood + [other_player_i, other_player_j]

        return state

    def get_action(self, state):
        # tradeoff exploration / exploitation
        final_action = [0, 0, 0, 0, 0, 0]
        if np.random.random() > self.epsilon:
            state = np.array(state)
            current_state = torch.tensor(state, dtype=torch.float).to(self.device)
            prediction = self.brain(current_state)
            action = torch.argmax(prediction).item()
            final_action[action] = 1
        else:
            if self.epsilon < 0.7:
                action = random.randint(0, 5)     #second exploration phase (eps < 0.7) experiment all controls 
            else:
                action = random.randint(0, 3)     #first explorarion phase (eps=[1.0 - 0.7]) is just moving around the map
            final_action[action] = 1

        return final_action
    
    #method to perform only model prediction in game
    def perform_action(self, state):
        # tradeoff exploration / exploitation
        final_action = [0, 0, 0, 0, 0, 0]
        
        state = np.array(state)
        current_state = torch.tensor(state, dtype=torch.float).to(self.device)
        prediction = self.brain(current_state)
        action = torch.argmax(prediction).item()
        final_action[action] = 1

        return final_action

    def remember(self, state, action, reward, next_state, gameover):
        self.memory.append((state, action, reward, next_state, gameover))

    def train_short_memory(self, state, action, reward, next_state, gameover):
        start = time.time()
        state = np.array(state)
        next_state = np.array(next_state)
        self.trainer.train_step(state, action, reward, next_state, gameover)
        end = time.time()

    def train_long_memory(self):
        if len(self.memory) < self.batch_size:
            return
        else:
            batch_sample = random.sample(self.memory, self.batch_size)

        states, actions, rewards, next_states, gameovers = zip(*batch_sample)
        #print('pre : ', type(states), len(states), len(states[0]),'\n')

        # convert to numpy arrays
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        gameovers = np.array(gameovers)

        self.trainer.train_step(states, actions, rewards, next_states, gameovers)

        self.brain.save()

        self.decrement_epsilon()

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

class Agent_alpha_8:
    def __init__(self, name='model', Qtrainer=QTrainer_beta_1, lr=0.001, batch_size=1000, max_memory=100000, epsilon = 1.0, eps_dec= 5e-4, eps_min = 0.05):
        self.agent_name = "alpha_8"
        self.name = name
        self.Qtrainer = Qtrainer
        self.lr = lr
        self.batch_size = batch_size
        self.max_memory = max_memory
        self.n_games = 0
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=self.max_memory)  # automatic popleft()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.brain = QNet([64, 512, 128, 6], self.agent_name, self.name).to(self.device)
        self.trainer = self.Qtrainer(self.brain, self.lr, self.gamma)

        if self.brain.load():
            print("Model loaded")

        print(f"AGENT ALPHA 8: training {self.name} with {self.device} device")

    def get_state(self, game, player):
        start = time.time()
        x = player.x
        y = player.y
        i = y // player.size
        j = x // player.size

        other_player = game.players[0] if player.obj_type == 'seeker' else game.players[1]
        other_player_i = other_player.y // other_player.size
        other_player_j = other_player.x // other_player.size

        objects = {'wall': '100000', 'floor': '010000', 'hider': '001000', 'movable_wall': '000100', 'seeker': '000010', 'map_edge': '000001'}

        #av = {'sx': False, 'dx': False, 'u':False, 'd':False}
        av = game.check_available_positions(player)
        av_pos = [int(av['u']), int(av['dx']), int(av['d']), int(av['sx'])]

        lidar = player.lidar
        lidar_data = []
        for elem in range(len(lidar)):
            for n in objects[lidar[elem][0]]:
                lidar_data.append(int(n))
            lidar_data.append(lidar[elem][1])
        #lidar_data = [0 0 0 0 0 1 7.56 , 100000 89.3 ...] (1-hot encoding obj_type + distance)

        state = [i,j] + av_pos + lidar_data + [other_player_i, other_player_j]

        return state

    def get_action(self, state):
        # tradeoff exploration / exploitation
        final_action = [0, 0, 0, 0, 0, 0]
        if np.random.random() > self.epsilon:
            state = np.array(state)
            current_state = torch.tensor(state, dtype=torch.float).to(self.device)
            prediction = self.brain(current_state)
            action = torch.argmax(prediction).item()
            final_action[action] = 1
        else:
            if self.epsilon < 0.7:
                action = random.randint(0, 5)     #second exploration phase (eps < 0.7) experiment all controls 
            else:
                action = random.randint(0, 3)     #first explorarion phase (eps=[1.0 - 0.7]) is just moving around the map
            final_action[action] = 1

        return final_action
     
    #method to perform only model prediction in game
    def perform_action(self, state):
        # tradeoff exploration / exploitation
        final_action = [0, 0, 0, 0, 0, 0]
        
        state = np.array(state)
        current_state = torch.tensor(state, dtype=torch.float).to(self.device)
        prediction = self.brain(current_state)
        action = torch.argmax(prediction).item()
        final_action[action] = 1

        return final_action

    def remember(self, state, action, reward, next_state, gameover):
        self.memory.append((state, action, reward, next_state, gameover))

    def train_short_memory(self, state, action, reward, next_state, gameover):
        start = time.time()
        state = np.array(state)
        next_state = np.array(next_state)
        self.trainer.train_step(state, action, reward, next_state, gameover)
        end = time.time()

    def train_long_memory(self):
        if len(self.memory) < self.batch_size:
            return
        else:
            batch_sample = random.sample(self.memory, self.batch_size)

        states, actions, rewards, next_states, gameovers = zip(*batch_sample)
        #print('pre : ', type(states), len(states), len(states[0]),'\n')

        # convert to numpy arrays
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        gameovers = np.array(gameovers)

        self.trainer.train_step(states, actions, rewards, next_states, gameovers)

        self.brain.save()

        self.decrement_epsilon()

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

# Agent Perfect_seeker is a cheater, never lend him your money, he uses no neural model or learn at all but in empty maps is really good at finding hiders
class Perfect_seeker_0:
    def __init__(self, name='model'):
        self.agent_name = "perfect_seeker_0"
        self.name = name
        self.n_games = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"AGENT PERFECT SEEKER 0: playing as {self.name}")

    # only uses game and other information to find perfect moves
    def get_state(self, game, player):
        return {"player": player, "other": game.players[0]}

    def get_action(self, state):
        # strategy: move towards the hider one step at a time
        other = state["other"]
        player = state["player"]

        # Perfect_seeker_0 -is dumb walls are a big problem for him
        chance = random.randint(0, 1)
        if chance == 0:
            # plan A:
            if other.x > player.x:
                return [0, 0, 0, 1, 0, 0]
            elif other.x < player.x:
                return [0, 0, 1, 0, 0, 0]
            elif other.y > player.y:
                return [0, 1, 0, 0, 0, 0]
            elif other.y < player.y:
                return [1, 0, 0, 0, 0, 0]
            else:
                return [0, 0, 0, 0, 0, 1]

        else:
            # plan B:
            if other.y > player.y:
                return [0, 1, 0, 0, 0, 0]
            elif other.y < player.y:
                return [1, 0, 0, 0, 0, 0]
            elif other.x > player.x:
                return [0, 0, 0, 1, 0, 0]
            elif other.x < player.x:
                return [0, 0, 1, 0, 0, 0]
            else:
                return [0, 0, 0, 0, 0, 1]

        # perfect seeker 0 never moves walls

# Agent Small_brain is an experiment, for the hider, he literally only knows if positions around him are available or not and the position of it's opponent + distance
class Small_brain_0:
    def __init__(self, name='model', Qtrainer=QTrainer_beta_1, lr=0.001, batch_size=1000, max_memory=100000, epsilon = 1.0, eps_dec= 5e-4, eps_min = 0.05):
        self.agent_name = "small_brain_0"
        self.name = name
        self.Qtrainer = Qtrainer
        self.lr = lr
        self.batch_size = batch_size
        self.max_memory = max_memory
        self.n_games = 0
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=self.max_memory)  # automatic popleft()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.brain = QNet([9, 32, 32, 6], self.agent_name, self.name).to(self.device)
        self.trainer = self.Qtrainer(self.brain, self.lr, self.gamma)

        if self.brain.load():
            print("Model loaded")

        print(f"AGENT ALPHA 8: training {self.name} with {self.device} device")

    def get_state(self, game, player):
        start = time.time()
        x = player.x
        y = player.y
        i = y // player.size
        j = x // player.size

        other_player = game.players[0] if player.obj_type == 'seeker' else game.players[1]
        other_player_i = other_player.y // other_player.size
        other_player_j = other_player.x // other_player.size

        av = game.check_available_positions(player)
        #av = {'sx': False, 'dx': False, 'u':False, 'd':False}
        
        av_pos = [int(av['u']), int(av['dx']), int(av['d']), int(av['sx'])]
        rel_distance = np.sqrt((other_player_i - i)**2 + (other_player_j - j)**2) / (game.rows + game.cols)

        sx = int(other_player_j - j <= 0)
        dx = int(other_player_j - j >= 0)
        u = int(other_player_i - i <= 0)
        d = int(other_player_i - i >= 0)

        state = av_pos + [sx, dx, u, d] + [rel_distance]

        return state

    def get_action(self, state):
        # tradeoff exploration / exploitation
        final_action = [0, 0, 0, 0, 0, 0]
        if np.random.random() > self.epsilon:
            state = np.array(state)
            current_state = torch.tensor(state, dtype=torch.float).to(self.device)
            prediction = self.brain(current_state)
            action = torch.argmax(prediction).item()
            final_action[action] = 1
        else:
            action = random.randint(0, 5)
            final_action[action] = 1

        return final_action
    
    #method to perform only model prediction in game
    def perform_action(self, state):
        # tradeoff exploration / exploitation
        final_action = [0, 0, 0, 0, 0, 0]
        
        state = np.array(state)
        current_state = torch.tensor(state, dtype=torch.float).to(self.device)
        prediction = self.brain(current_state)
        action = torch.argmax(prediction).item()
        final_action[action] = 1

        return final_action

    def remember(self, state, action, reward, next_state, gameover):
        self.memory.append((state, action, reward, next_state, gameover))

    def train_short_memory(self, state, action, reward, next_state, gameover):
        start = time.time()
        state = np.array(state)
        next_state = np.array(next_state)
        self.trainer.train_step(state, action, reward, next_state, gameover)
        end = time.time()

    def train_long_memory(self):
        if len(self.memory) < self.batch_size:
            return
        else:
            batch_sample = random.sample(self.memory, self.batch_size)

        states, actions, rewards, next_states, gameovers = zip(*batch_sample)
        #print('pre : ', type(states), len(states), len(states[0]),'\n')

        # convert to numpy arrays
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        gameovers = np.array(gameovers)

        self.trainer.train_step(states, actions, rewards, next_states, gameovers)

        self.brain.save()

        self.decrement_epsilon()

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
