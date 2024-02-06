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

    # Load the replay memory SELECTIVELY from the file (when the function is called the file already exists)
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

        # Brain designed for a 26x26 map
        self.brain = ConvQNet([[1, 3, 3, 1, 1], [3, 1, 9, 1, 0], [1, 1, 7, 1, 0], [1, 1, 5, 1, 0]], [64, 128, 128, 6], self.agent_name, self.name).to(self.device)
        # Convolutional layers definition is not flexible and needs a coherent input (i.e. Map size dependent)

        self.trainer = self.Qtrainer(self.brain, self.lr, self.gamma, convolutional=True)

        # Load the model if it exists
        if self.brain.load():
            print("Model loaded")

        print(f"AGENT HIVEMIND: training {self.name} with {self.device} device")

    # Load the memory from the file if it exists   
    def init_memory(self):
        # Check if memory file exists
        if not os.path.exists("./hivemind/memory/" + self.name +".txt"):
            # If directory already exists, return
            if os.path.exists("./hivemind/memory"):
                return
            # Otherwise create the directory
            else:
                os.makedirs("./hivemind/memory")
                return
        # Recall last MAX_MEMORY lines from the memory file
        with open("./hivemind/memory/" + self.name +".txt", "r") as f:
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

    # Load the replay memory SELECTIVELY from the file (when the function is called the file already exists)
    def load_replay_memory(self, criterion="reward"):
        with open("./hivemind/memory/" + self.name +".txt", "r") as f:
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

        # Load up to batch_size lines - Use ast to easily parse the matrix representation of the states
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

    # Get the state of the game for the agent
    def get_state(self, game, player):

        # Encoding of all the object types - None is either outside the map or out of the player's view (hidden by an object) - Player.mask_view()
        objects = {'wall': '5', 'floor': '1', 'hider': '100', 'movable_wall': '10','seeker': '-100', None: '0'}

        # Encoding of the whole map
        state = []
        for row in range(len(player.map)):
            state.append([])
            for cell in range(len(player.map[row])):
                state[row].append(int(objects[game.map[row][cell].obj_type]))

        return state

    # Pick the next action to take - Tradeoff exploration / exploitation
    def get_action(self, state):

        # Final action is a one-hot encoded vector
        final_action = [0,0,0,0,0,0]

        # Move randomly
        if random.randint(0, 200) < self.epsilon:   # 200 is arbitrary
            action = random.randint(0, 5)
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
        
        with open("./hivemind/memory/" + self.name +".txt", "a") as f:
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

        file_path = "./hivemind/memory/" + self.name + ".txt"
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

# Agent Beta doesn't employ replay or file memory anymore, the exploration is now fixed to a minimum reached after a certain number of games:
    # Agent Beta 0 is the first prototype that doesn't rely on the file memory and uses player's view and neighbourhood
    # Agent Beta 1, with the same encoding of Beta 0, uses more information about player's positions, distances and directions
    # Agent Beta 2 mimicks Beta 1 but one-hot encodes the direction and doesn't rely on the distance, resulting in the biggest input size of all the Beta agents, aided by the bigger network
    # Agent Beta 3 doesn't use the view of the players and only relies on the neighbourhood and positions, with the smallest input size of the series, it keeps the deep architecture of Beta 2 (with fewer neurons per layer)
    # Agent Beta 4 is the only one to employ the lidar vision system paired with positions and available actions, no longer relying on the player's view and neighbourhood
class Agent_beta:
    def __init__(self, beta=0, name='model', Qtrainer=QTrainer_beta_1, lr=0.0005, batch_size=1000, max_memory=100000, eps_dec= 5e-4, eps_min = 0.01):
        # Beta generation number (0,1,2,3,4)
        self.beta = beta
        # Agent name corresponds to the beta generation and is used to save and load the model, configs and memory
        self.agent_name = "beta_"+str(self.beta)
        # Seeker or Hider
        self.name = name
        # Q_trainer class is instantiated without parameters to include it in the config file
        self.Qtrainer = Qtrainer

        # Agent hyperparameters
        self.lr = lr
        self.batch_size = batch_size
        self.max_memory = max_memory
        self.n_games = 0        # number of games played
        self.epsilon = 1.0      # randomness
        self.eps_dec = eps_dec  # epsilon decrement
        self.eps_min = eps_min  # minimum epsilon (to keep some degree of exploration)
        self.gamma = 0.9        # future expected reward discount rate

        # Agent long term memory
        self.memory = deque(maxlen=self.max_memory)         # agent memory, queue with maxlen to automatically pop left 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # beta_generation picks the respective network architecture for the agent
        beta_generation = {0: [75, 256, 256, 6], 1: [81, 256, 256, 6], 2: [83, 256, 128, 128, 256, 6], 3: [28, 256, 128, 64, 32, 6], 4: [64, 512, 128, 6]}
        self.brain = QNet(beta_generation[self.beta], self.agent_name, self.name).to(self.device)
        self.trainer = self.Qtrainer(self.brain, self.lr, self.gamma)

        # Load the model if it exists
        if self.brain.load():
            print("Model loaded")

        print(f"AGENT BETA " + str(self.beta) + f": training {self.name} with {self.device} device")

    # Get the state of the game for the agent - Using different encodings/data for different beta generations
    def get_state(self, game, player):
        x = player.x
        y = player.y

        i = y // player.size
        j = x // player.size

        other_player = game.players[0] if player.obj_type == 'seeker' else game.players[1]
        other_player_i = other_player.y // other_player.size
        other_player_j = other_player.x // other_player.size

        # Beta 0
        if self.beta == 0:

            # One-hot encoding of all the object types - None is either outside the map or out of the player's view (hidden by an object) - Player.mask_view()
            objects = {'wall': '10000', 'floor': '01000', 'hider': '00100', 'movable_wall': '00010', 'seeker': '00001', None: '00000'}

            # What the player sees according to policy
            view = player.view

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
            neighbourhood = []  # order - left, back, right

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
        # Beta 1   
        elif self.beta == 1:
            
            # Encoding of all the object types - None is either outside the map or out of the player's view (hidden by an object) - Player.mask_view()
            objects = {'wall': '10000', 'floor': '01000', 'hider': '00100', 'movable_wall': '00010', 'seeker': '00001', None: '00000'}

            # Euclidean distance from the other player
            distance = np.sqrt((other_player_i - i)**2 + (other_player_j - j)**2)
            # Normalize the distance
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

            # What the player sees according to policy
            view = player.view
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
            neighbourhood = []  # order - left back right
            
            if player.direction == 'u':     # up
                left = player.map[i][j - 1].obj_type if j - 1 >= 0 else None
                back = player.map[i + 1][j].obj_type if i + 1 < game.rows else None
                right = player.map[i][j + 1].obj_type if j + 1 < game.cols else None

            elif player.direction == 'd':   # down
                left = player.map[i][j + 1].obj_type if j + 1 < game.cols else None
                back = player.map[i - 1][j].obj_type if i - 1 >= 0 else None
                right = player.map[i][j - 1].obj_type if j - 1 >= 0 else None

            elif player.direction == 'l':   # left
                left = player.map[i + 1][j].obj_type if i + 1 < game.rows else None
                back = player.map[i][j + 1].obj_type if j + 1 < game.cols else None
                right = player.map[i - 1][j].obj_type if i - 1 >= 0 else None

            elif player.direction == 'r':   # right
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

            state = [i,j] + view_vector + neighbourhood + [distance, direction] + [other_player_i, other_player_j]
        # Beta 2
        elif self.beta == 2:
            # One-hot encoding of the directions
            directions = {'u' : '0001', 'd' : '0010', 'l' : '0100', 'r' : '1000'}
            # One-hot encoding of all the object types - None is either outside the map or out of the player's view (hidden by an object) - Player.mask_view()
            objects = {'wall': '10000', 'floor': '01000', 'hider': '00100', 'movable_wall': '00010', 'seeker': '00001', None: '00000'}
            
            # What the player sees according to policy
            view = player.view

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
            neighbourhood = []  # order - left back right
            
            # None is outside of the map
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

            # Encode the neighbourhood
            for n in objects[left]:
                neighbourhood.append(int(n))
            for n in objects[back]:
                neighbourhood.append(int(n))
            for n in objects[right]:
                neighbourhood.append(int(n))

            direction = []

            # Encode direction
            for n in directions[player.direction]:
                direction.append(int(n))


            state = [i,j] + view_vector + neighbourhood + direction + [other_player_i, other_player_j]
        # Beta 3
        elif self.beta == 3:
            
            # One-hot encoding of all the object types - map_edge is outside the map - No confilts with view masking as it's not used
            objects = {'wall': '100000', 'floor': '010000', 'hider': '001000', 'movable_wall': '000100', 'seeker': '000010', 'map_edge': '000001'}
            
            # The neighbourhood is composed of the 4 cells around the player
            neighbourhood = []  # order - left back right up

            left = player.map[i][j - 1].obj_type if j - 1 >= 0 else 'map_edge'
            back = player.map[i + 1][j].obj_type if i + 1 < game.rows else 'map_edge'
            right = player.map[i][j + 1].obj_type if j + 1 < game.cols else 'map_edge'
            up = player.map[i-1][j].obj_type if i >= 1 else 'map_edge'

            # Encode the neighbourhood
            for n in objects[left]:
                neighbourhood.append(int(n))
            for n in objects[back]:
                neighbourhood.append(int(n))
            for n in objects[right]:
                neighbourhood.append(int(n))
            for n in objects[up]:
                neighbourhood.append(int(n))

            state = [i,j] + neighbourhood + [other_player_i, other_player_j]
        # Beta 4
        elif self.beta == 4:
            # One-hot encoding of all the object types - map_edge is outside the map - No confilts with view masking as it's not used
            objects = {'wall': '100000', 'floor': '010000', 'hider': '001000', 'movable_wall': '000100', 'seeker': '000010', 'map_edge': '000001'}

            # Available positions for the next taken action - {'sx': False, 'dx': False, 'u':False, 'd':False}
            av = game.check_available_positions(player)
            av_pos = [int(av['u']), int(av['dx']), int(av['d']), int(av['sx'])]

            # What the 8 lidar sensors detect
            lidar_data = []

            # Encode the lidar data
            for elem in range(len(player.lidar)):
                for n in objects[player.lidar[elem][0]]:
                    lidar_data.append(int(n))
                lidar_data.append(player.lidar[elem][1])
            #lidar_data = [0,0,0,0,0,1,7.56 ; 1,0,0,0,0,0,89.3 ; ...] (1-hot encoded obj_type + distance)

            state = [i,j] + av_pos + lidar_data + [other_player_i, other_player_j]
            
        return state
    
    # Pick the next action to take - Tradeoff exploration / exploitation
    def get_action(self, state):
        # Final action is a one-hot encoded vector
        final_action = [0, 0, 0, 0, 0, 0]

        # Move randomly
        if np.random.random() > self.epsilon:
            state = np.array(state)
            current_state = torch.tensor(state, dtype=torch.float).to(self.device)
            prediction = self.brain(current_state)
            action = torch.argmax(prediction).item()
            final_action[action] = 1
        
        # Move according to the policy network
        else:
            action = random.randint(0, 5)
            final_action[action] = 1

        return final_action

    # Store the experience in the agent's memory
    def remember(self, state, action, reward, next_state, gameover):
        self.memory.append((state, action, reward, next_state, gameover))

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
        # If memory is not big enough, skip training
        if len(self.memory) < self.batch_size:
            return
        # Sample a random batch from the memory
        else:
            batch_sample = random.sample(self.memory, self.batch_size)

        # Separate the batch into its components
        states, actions, rewards, next_states, gameovers = zip(*batch_sample)

        # Cast to numpy arrays for the torch.tensor conversion
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        gameovers = np.array(gameovers)

        self.trainer.train_step(states, actions, rewards, next_states, gameovers)

        # Save the model
        self.brain.save()

        # Decrease epsilon
        self.decrement_epsilon()

        # Log time took for training
        end = time.time()
        print(f"\033[92mtraning long memory took: {end - start} seconds\033[0m")

    # Decrease epsilon
    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

# Agent Perfect_seeker is a cheater, never lend him your money, he uses no neural model or learn at all but in empty maps is really good at finding hiders
    # It's used to test learning with other agents on the Hider side
class Perfect_seeker:
    def __init__(self, name='model'):

        self.agent_name = "perfect_seeker"
        self.name = name
        self.n_games = 0        # Number of games played
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"AGENT PERFECT SEEKER: playing as {self.name}")

    # Only uses game and other information to find perfect moves (in empy maps)
    def get_state(self, game, player):
        return {"player": player, "other": game.players[0]}

    # Based on relative position take an action to reach your opponent
    def get_action(self, state):
        # Strategy: move towards the hider one step at a time
        other = state["other"]
        player = state["player"]

        # Perfect_seeker is dumb, walls are a big problem for him

        # In order to achieve a certain degree of randomness, the order of the moves is reversed in the two plans A and B, with 50% chance of choosing one or the other
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

        # Perfect seeker never moves walls

# Agent Small_brain is an experiment for the hider, he literally only knows if positions around him are available or not and if it's opponent is on a certain side plus the distance
class Small_brain:
    def __init__(self, name='model', Qtrainer=QTrainer_beta_1, lr=0.001, batch_size=1000, max_memory=100000, epsilon = 1.0, eps_dec= 5e-4, eps_min = 0.05):
        self.agent_name = "small_brain"
        # Seeker or Hider
        self.name = name
        # Qtrainer class is instantiated without parameters to include it in the config file
        self.Qtrainer = Qtrainer
        # Agent hyperparameters
        self.lr = lr
        self.batch_size = batch_size
        self.max_memory = max_memory
        self.n_games = 0        # number of games played
        self.epsilon = epsilon      # randomness
        self.eps_dec = eps_dec      # epsilon decrement
        self.eps_min = eps_min      # minimum epsilon (to keep some degree of exploration)
        self.gamma = 0.9        # future expected reward discount rate
        self.memory = deque(maxlen=self.max_memory)     # agent memory, queue with maxlen to automatically popleft
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.brain = QNet([9, 32, 32, 6], self.agent_name, self.name).to(self.device)
        self.trainer = self.Qtrainer(self.brain, self.lr, self.gamma)
        # Load the model if it exists
        if self.brain.load():
            print("Model loaded")

        print(f"AGENT SMALL BRAIN: training {self.name} with {self.device} device")

    # Get the state of the game for the agent 
    def get_state(self, game, player):
        x = player.x
        y = player.y
        i = y // player.size
        j = x // player.size

        other_player = game.players[0] if player.obj_type == 'seeker' else game.players[1]
        other_player_i = other_player.y // other_player.size
        other_player_j = other_player.x // other_player.size

        # Available positions for the next taken action - {'sx': False, 'dx': False, 'u':False, 'd':False}
        av = game.check_available_positions(player)
        
        # Encode the available positions
        av_pos = [int(av['u']), int(av['dx']), int(av['d']), int(av['sx'])]
        rel_distance = np.sqrt((other_player_i - i)**2 + (other_player_j - j)**2) / (game.rows + game.cols)
        
        sx = int(other_player_j - j <= 0)
        dx = int(other_player_j - j >= 0)
        u = int(other_player_i - i <= 0)
        d = int(other_player_i - i >= 0)

        state = av_pos + [sx, dx, u, d] + [rel_distance]

        return state

    # Pick the next action to take
    def get_action(self, state):
        # Final action is one-hot encoded vector
        final_action = [0, 0, 0, 0, 0, 0]
        # Move randomly
        if np.random.random() > self.epsilon:
            state = np.array(state)
            current_state = torch.tensor(state, dtype=torch.float).to(self.device)
            prediction = self.brain(current_state)
            action = torch.argmax(prediction).item()
            final_action[action] = 1
        # Moving according to the policy network
        else:
            action = random.randint(0, 5)
            final_action[action] = 1

        return final_action
    
    # Store the experience in the agent's memory
    def remember(self, state, action, reward, next_state, gameover):
        self.memory.append((state, action, reward, next_state, gameover))

    # Online training - short term memory
    def train_short_memory(self, state, action, reward, next_state, gameover):
        # Cast numpy arrays for the torch.tensor conversion 
        state = np.array(state)
        next_state = np.array(next_state)
        action = np.array(action)
        self.trainer.train_step(state, action, reward, next_state, gameover)

    # Batch training - long term memory
    def train_long_memory(self):
        # if the memory is not big enough, skip training 
        if len(self.memory) < self.batch_size:
            return
        # Sample a random batch from the memory
        else:
            batch_sample = random.sample(self.memory, self.batch_size)

        # Separate the batch into its components
        states, actions, rewards, next_states, gameovers = zip(*batch_sample)

        # Cast numpy arrays for the torch.tensor conversion 
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        gameovers = np.array(gameovers)

        self.trainer.train_step(states, actions, rewards, next_states, gameovers)

        # Save the model
        self.brain.save()

        # Decrease the epsilon
        self.decrement_epsilon()

    # Decrease epsilon
    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
