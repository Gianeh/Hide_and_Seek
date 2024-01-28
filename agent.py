import torch
import random
from collections import deque
import numpy as np
from models import QTrainer, QTrainer_beta_1, QNet, ConvQNet
import os
import ast # to easily load memory

# import time for probing purposes
import time

#Default parameters for all models
MAX_MEMORY = 100000
BATCH_SIZE = 1000
LR = 0.0005


# Agent Alpha 0 is the first complete prototype, every piece work but learning seems capped by input encoding or other unknown factors
class Agent_alpha_0:
    def __init__(self, name='model', Qtrainer=QTrainer, lr = LR, batch_size = BATCH_SIZE, max_memory = MAX_MEMORY):
        self.agent_name = "alpha_0"
        self.name = name
        self.Qtrainer = Qtrainer
        self.lr = lr
        self.batch_size = batch_size
        self.max_memory = max_memory
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.randomness = 200
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # automatic popleft()
        self.init_memory() # reload all previous memeories up to MAX_MEMORY
        self.replay_memory = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.brain = QNet([79, 256, 512, 256, 5], self.agent_name, self.name).to(self.device)
        self.trainer = self.Qtrainer(self.brain, self.lr, self.gamma)

        if self.brain.load():
            print("Model loaded")

        
    def init_memory(self):
        # check if memory file exists
        if not os.path.exists("./alpha_0/memory/" + self.name +".txt"):
            if os.path.exists("./alpha_0/memory"):
                return
            else:
                os.makedirs("./alpha_0/memory")
                return
        # recall last lines of memory up to MAX_MEMORY
        with open("./alpha_0/memory/" + self.name +".txt", "r") as f:
            lines = f.readlines()
            if len(lines) > MAX_MEMORY:
                lines = lines[-MAX_MEMORY:]
            for line in lines:
                state, action, reward, next_state, gameover = line.split(";")
                state = np.array(state[1:-1].split(","), dtype=np.float32)
                action = np.array(action[1:-1].split(","), dtype=np.float32)
                reward = np.array(float(reward), dtype=np.float32)
                next_state = np.array(next_state[1:-1].split(","), dtype=np.float32)
                gameover = np.array(gameover == 'True')
                self.memory.append((state, action, reward, next_state, gameover))

    def load_replay_memory(self, criterion="reward"):
        with open("./alpha_0/memory/" + self.name +".txt", "r") as f:
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
            state = np.array(state[1:-1].split(","), dtype=np.float32)
            action = np.array(action[1:-1].split(","), dtype=np.float32)
            reward = np.array(float(reward), dtype=np.float32)
            next_state = np.array(next_state[1:-1].split(","), dtype=np.float32)
            gameover = np.array(gameover == 'True')
            self.replay_memory.append((state, action, reward, next_state, gameover))
                

    def get_state(self, game, player):
        start = time.time()
        x = player.x
        y = player.y
        x_norm = player.x / (game.width - player.size)
        y_norm = player.y / (game.height - player.size)
        # quadrati adiacenti in base alla direction del player
        view = player.view
        objects = {'wall': '10000', 'floor': '01000', 'hider': '00100', 'movable_wall': '00010','seeker': '00001', None: '00000'}
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

        neighbourhood = [] # left back right
        i = y // player.size
        j = x // player.size
        if player.direction == 'u':
            left = player.map[i][j-1].obj_type if j-1 >= 0 else None
            back = player.map[i+1][j].obj_type if i+1 < game.rows else None
            right = player.map[i][j+1].obj_type if j+1 < game.cols else None

        elif player.direction == 'd':
            left = player.map[i][j+1].obj_type if j+1 < game.cols else None
            back = player.map[i-1][j].obj_type if i-1 >= 0 else None
            right = player.map[i][j-1].obj_type if j-1 >= 0 else None

        elif player.direction == 'l':
            left = player.map[i+1][j].obj_type if i+1 < game.rows else None
            back = player.map[i][j+1].obj_type if j+1 < game.cols else None
            right = player.map[i-1][j].obj_type if i-1 >= 0 else None

        elif player.direction == 'r':
            left = player.map[i-1][j].obj_type if i-1 >= 0 else None
            back = player.map[i][j-1].obj_type if j-1 >= 0 else None
            right = player.map[i+1][j].obj_type if i+1 < game.rows else None
        
        for n in objects[left]:
            neighbourhood.append(int(n))
        for n in objects[back]:
            neighbourhood.append(int(n))
        for n in objects[right]:
            neighbourhood.append(int(n))

        # as a test, acquire other player's position
        # and normalize it
        other_player = game.players[0] if player.obj_type == 'seeker' else game.players[1]
        other_player_x = other_player.x / (game.width - other_player.size)
        other_player_y = other_player.y / (game.height - other_player.size)


        state = [x_norm,y_norm] + view_vector + neighbourhood + [other_player_x, other_player_y]

        end = time.time()
        #print("get_state: ", end - start)

        return state

    def get_action(self, state):
        start = time.time()
        # tradeoff exploration / exploitation
        self.epsilon = self.randomness - self.n_games    # 80 is arbitrary
        final_action = [0,0,0,0,0]
        if random.randint(0, 200) < self.epsilon:   # 200 is arbitrary
            action = random.randint(0, 4)
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
        with open("./alpha_0/memory/" + self.name +".txt", "a") as f:
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
            ''' Loss of the exploration phase in the long run
            batch_sample = []
            for i in range(len(self.memory)-1, len(self.memory) - 1 - BATCH_SIZE, -1):
                batch_sample.append(self.memory[i])
            '''
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
        file_path = "./alpha_0/memory/" + self.name + ".txt"
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
    
# Agent Alpha 1 is the second complete prototype, the aim is to improve the input encoding narrowing the possible existing problems with alpha 0
class Agent_alpha_1:
    def __init__(self, name='model', Qtrainer=QTrainer, lr = LR, batch_size = BATCH_SIZE, max_memory = MAX_MEMORY):
        self.agent_name = "alpha_1"
        self.name = name
        self.Qtrainer = Qtrainer
        self.lr = lr
        self.batch_size = batch_size
        self.max_memory = max_memory
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.randomness = 80
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=self.max_memory) # automatic popleft()
        self.init_memory() # reload all previous memeories up to MAX_MEMORY
        self.replay_memory = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.brain = QNet([19, 128, 128, 6], self.agent_name, self.name).to(self.device)
        self.trainer = self.Qtrainer(self.brain, self.lr, self.gamma)

        if self.brain.load():
            print("Model loaded")

        print(f"AGENT ALPHA 1: training {self.name} with {self.device} device")
        
    def init_memory(self):
        # check if memory file exists
        if not os.path.exists("./alpha_1/memory/" + self.name +".txt"):
            if os.path.exists("./alpha_1/memory"):
                return
            else:
                os.makedirs("./alpha_1/memory")
                return
        # recall last lines of memory up to MAX_MEMORY
        with open("./alpha_1/memory/" + self.name +".txt", "r") as f:
            lines = f.readlines()
            if len(lines) > self.max_memory:
                lines = lines[-self.max_memory:]
            for line in lines:
                state, action, reward, next_state, gameover = line.split(";")
                state = np.array(state[1:-1].split(","), dtype=np.float32)
                action = np.array(action[1:-1].split(","), dtype=np.float32)
                reward = np.array(float(reward), dtype=np.float32)
                next_state = np.array(next_state[1:-1].split(","), dtype=np.float32)
                gameover = np.array(gameover == 'True')
                self.memory.append((state, action, reward, next_state, gameover))

    def load_replay_memory(self, criterion="reward"):
        with open("./alpha_1/memory/" + self.name +".txt", "r") as f:
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
            state = np.array(state[1:-1].split(","), dtype=np.float32)
            action = np.array(action[1:-1].split(","), dtype=np.float32)
            reward = np.array(float(reward), dtype=np.float32)
            next_state = np.array(next_state[1:-1].split(","), dtype=np.float32)
            gameover = np.array(gameover == 'True')
            self.replay_memory.append((state, action, reward, next_state, gameover))
                
    # difference with alpha 0, the state is now encoded in a much simpler way
    def get_state(self, game, player):
        start = time.time()
        x = player.x
        y = player.y
        i = y // player.size
        j = x // player.size

        view = player.view
        objects = {'wall': '5', 'floor': '1', 'hider': '100', 'movable_wall': '10','seeker': '100', None: '0'}
        # ["wall", "floor", "hider", "movable_wall", None]
        view_vector = []
        for l in view:
            for c in l:
                if c is None:
                    view_vector.append(int(objects[c]))
                else:
                    view_vector.append(int(objects[c.obj_type]))

        neighbourhood = [] # left back right

        if player.direction == 'u':
            left = player.map[i][j-1].obj_type if j-1 >= 0 else None
            back = player.map[i+1][j].obj_type if i+1 < game.rows else None
            right = player.map[i][j+1].obj_type if j+1 < game.cols else None

        elif player.direction == 'd':
            left = player.map[i][j+1].obj_type if j+1 < game.cols else None
            back = player.map[i-1][j].obj_type if i-1 >= 0 else None
            right = player.map[i][j-1].obj_type if j-1 >= 0 else None

        elif player.direction == 'l':
            left = player.map[i+1][j].obj_type if i+1 < game.rows else None
            back = player.map[i][j+1].obj_type if j+1 < game.cols else None
            right = player.map[i-1][j].obj_type if i-1 >= 0 else None

        elif player.direction == 'r':
            left = player.map[i-1][j].obj_type if i-1 >= 0 else None
            back = player.map[i][j-1].obj_type if j-1 >= 0 else None
            right = player.map[i+1][j].obj_type if i+1 < game.rows else None
        
        neighbourhood.append(int(objects[left]))
        neighbourhood.append(int(objects[back]))
        neighbourhood.append(int(objects[right]))

        # as a test, acquire other player's position
        # and normalize it
        other_player = game.players[0] if player.obj_type == 'seeker' else game.players[1]
        other_player_i = other_player.y // other_player.size
        other_player_j = other_player.x // other_player.size


        state = [i,j] + view_vector + neighbourhood + [other_player_i, other_player_j]

        end = time.time()
        #print("get_state: ", end - start)

        return state

    def get_action(self, state):
        start = time.time()
        # tradeoff exploration / exploitation
        self.epsilon = self.randomness - self.n_games//3    # 80 is arbitrary --> //3 means we explore much more time with constant randomness probability!!
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
        with open("./alpha_1/memory/" + self.name +".txt", "a") as f:
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
            ''' Loss of the exploration phase in the long run
            batch_sample = []
            for i in range(len(self.memory)-1, len(self.memory) - 1 - BATCH_SIZE, -1):
                batch_sample.append(self.memory[i])
            '''
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
        file_path = "./alpha_1/memory/" + self.name + ".txt"
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

# Agent Alpha 2, comes to life after considerations made on hivemind prototypes, what about mimicking other examples of the same game? the state space is completely changed again in advantage of a coordinate/distance representation
class Agent_alpha_2:
    def __init__(self, name='model', Qtrainer=QTrainer, lr = LR, batch_size = BATCH_SIZE, max_memory = MAX_MEMORY):
        self.agent_name = "alpha_2"
        self.name = name
        self.Qtrainer = Qtrainer
        self.lr = lr
        self.batch_size = batch_size
        self.max_memory = max_memory
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.randomness = 80
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=self.max_memory) # automatic popleft()
        self.init_memory() # reload all previous memeories up to MAX_MEMORY
        self.replay_memory = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.brain = QNet([21, 256, 36, 6], self.agent_name, self.name).to(self.device)
        self.trainer = self.Qtrainer(self.brain, self.lr, self.gamma)

        if self.brain.load():
            print("Model loaded")

        print(f"AGENT ALPHA 2: training {self.name} with {self.device} device")
        
    def init_memory(self):
        # check if memory file exists
        if not os.path.exists("./alpha_2/memory/" + self.name +".txt"):
            if os.path.exists("./alpha_2/memory"):
                return
            else:
                os.makedirs("./alpha_2/memory")
                return
        # recall last lines of memory up to MAX_MEMORY
        with open("./alpha_2/memory/" + self.name +".txt", "r") as f:
            lines = f.readlines()
            if len(lines) > self.max_memory:
                lines = lines[-self.max_memory:]
            for line in lines:
                state, action, reward, next_state, gameover = line.split(";")
                state = np.array(state[1:-1].split(","), dtype=np.float32)
                action = np.array(action[1:-1].split(","), dtype=np.float32)
                reward = np.array(float(reward), dtype=np.float32)
                next_state = np.array(next_state[1:-1].split(","), dtype=np.float32)
                gameover = np.array(gameover == 'True')
                self.memory.append((state, action, reward, next_state, gameover))

    def load_replay_memory(self, criterion="reward"):
        with open("./alpha_2/memory/" + self.name +".txt", "r") as f:
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
            state = np.array(state[1:-1].split(","), dtype=np.float32)
            action = np.array(action[1:-1].split(","), dtype=np.float32)
            reward = np.array(float(reward), dtype=np.float32)
            next_state = np.array(next_state[1:-1].split(","), dtype=np.float32)
            gameover = np.array(gameover == 'True')
            self.replay_memory.append((state, action, reward, next_state, gameover))

    # difference with alpha 0, the state is now encoded in a much simpler way
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

        objects = {'wall': '5', 'floor': '1', 'hider': '100', 'movable_wall': '10','seeker': '100', None: '0'}
        view = player.view
        view_vector = []
        for l in view:
            for c in l:
                if c is None:
                    view_vector.append(int(objects[c]))
                else:
                    view_vector.append(int(objects[c.obj_type])/100)

        neighbourhood = [] # left back right

        if player.direction == 'u':
            left = player.map[i][j-1].obj_type if j-1 >= 0 else None
            back = player.map[i+1][j].obj_type if i+1 < game.rows else None
            right = player.map[i][j+1].obj_type if j+1 < game.cols else None

        elif player.direction == 'd':
            left = player.map[i][j+1].obj_type if j+1 < game.cols else None
            back = player.map[i-1][j].obj_type if i-1 >= 0 else None
            right = player.map[i][j-1].obj_type if j-1 >= 0 else None

        elif player.direction == 'l':
            left = player.map[i+1][j].obj_type if i+1 < game.rows else None
            back = player.map[i][j+1].obj_type if j+1 < game.cols else None
            right = player.map[i-1][j].obj_type if i-1 >= 0 else None

        elif player.direction == 'r':
            left = player.map[i-1][j].obj_type if i-1 >= 0 else None
            back = player.map[i][j-1].obj_type if j-1 >= 0 else None
            right = player.map[i+1][j].obj_type if i+1 < game.rows else None

        for n in objects[left]:
            neighbourhood.append(int(n)/100)
        for n in objects[back]:
            neighbourhood.append(int(n)/100)
        for n in objects[right]:
            neighbourhood.append(int(n)/100)
        
        # normalize everything
        i = i / game.rows
        j = j / game.cols
        distance = distance / (game.rows + game.cols)
        other_player_i = other_player_i / game.rows
        other_player_j = other_player_j / game.cols


            

        state = [i,j] + view_vector + neighbourhood + [distance, direction] + [other_player_i, other_player_j]

        end = time.time()
        #print("get_state: ", end - start)

        return state


    def get_action(self, state):
        start = time.time()
        # tradeoff exploration / exploitation
        self.epsilon = self.randomness - self.n_games//3    # 80 is arbitrary --> //3 means we explore much more time with constant randomness probability!!
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
        with open("./alpha_2/memory/" + self.name +".txt", "a") as f:
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
            ''' Loss of the exploration phase in the long run
            batch_sample = []
            for i in range(len(self.memory)-1, len(self.memory) - 1 - BATCH_SIZE, -1):
                batch_sample.append(self.memory[i])
            '''
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
        file_path = "./alpha_2/memory/" + self.name + ".txt"
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

# Hivemind series, as the name suggests, is meant to be able to predict next action taking into account the whole map
class Agent_hivemind_0:
    def __init__(self, name='model', Qtrainer=QTrainer, lr = LR, batch_size = BATCH_SIZE, max_memory = MAX_MEMORY):
        self.agent_name = "hivemind_0"
        self.name = name
        self.Qtrainer = Qtrainer
        self.lr = lr
        self.batch_size = batch_size
        self.max_memory = max_memory
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.randomness = 200
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=self.max_memory) # automatic popleft()
        self.init_memory() # reload all previous memories up to MAX_MEMORY
        self.replay_memory = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.brain = ConvQNet([[1, 3, 3, 1, 1], [3, 1, 7, 1, 0]], [36, 128, 128, 6], self.agent_name, self.name).to(self.device)
        self.trainer = self.Qtrainer(self.brain, self.lr, self.gamma, convolutional=True)

        if self.brain.load():
            print("Model loaded")

        print(f"AGENT HIVEMIND 0: training {self.name} with {self.device} device")
        
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
            ''' Loss of the exploration phase in the long run
            batch_sample = []
            for i in range(len(self.memory)-1, len(self.memory) - 1 - BATCH_SIZE, -1):
                batch_sample.append(self.memory[i])
            '''
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



class Agent_alpha_3:
    def __init__(self, name='model', Qtrainer=QTrainer, lr=LR, batch_size=BATCH_SIZE, max_memory=MAX_MEMORY):
        self.agent_name = "alpha_3"
        self.name = name
        self.Qtrainer = Qtrainer
        self.lr = lr
        self.batch_size = batch_size
        self.max_memory = max_memory
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.randomness = 200
        self.gamma = 0.7  # discount rate
        self.memory = deque(maxlen=self.max_memory)  # automatic popleft()
        self.init_memory()  # reload all previous memeories up to MAX_MEMORY
        self.replay_memory = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.brain = QNet([75, 128, 256, 256, 128, 6], self.agent_name, self.name).to(self.device)
        self.trainer = self.Qtrainer(self.brain, self.lr, self.gamma)

        if self.brain.load():
            print("Model loaded")

        print(f"AGENT ALPHA 3: training {self.name} with {self.device} device")

    def init_memory(self):
        # check if memory file exists
        if not os.path.exists("./alpha_3/memory/" + self.name + ".txt"):
            if os.path.exists("./alpha_3/memory"):
                return
            else:
                os.makedirs("./alpha_3/memory")
                return
        # recall last lines of memory up to MAX_MEMORY
        with open("./alpha_3/memory/" + self.name + ".txt", "r") as f:
            lines = f.readlines()
            if len(lines) > self.max_memory:
                lines = lines[-self.max_memory:]
            for line in lines:
                state, action, reward, next_state, gameover = line.split(";")
                state = np.array(state[1:-1].split(","), dtype=np.float32)
                action = np.array(action[1:-1].split(","), dtype=np.float32)
                reward = np.array(float(reward), dtype=np.float32)
                next_state = np.array(next_state[1:-1].split(","), dtype=np.float32)
                gameover = np.array(gameover == 'True')
                self.memory.append((state, action, reward, next_state, gameover))

    def load_replay_memory(self, criterion="reward"):
        with open("./alpha_3/memory/" + self.name + ".txt", "r") as f:
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
            state = np.array(state[1:-1].split(","), dtype=np.float32)
            action = np.array(action[1:-1].split(","), dtype=np.float32)
            reward = np.array(float(reward), dtype=np.float32)
            next_state = np.array(next_state[1:-1].split(","), dtype=np.float32)
            gameover = np.array(gameover == 'True')
            self.replay_memory.append((state, action, reward, next_state, gameover))

    # difference with alpha 0, the state is now encoded in a much simpler way
    def get_state(self, game, player):
        start = time.time()
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

        end = time.time()
        # print("get_state: ", end - start)

        return state

    def get_action(self, state):
        start = time.time()
        # tradeoff exploration / exploitation
        self.epsilon = self.randomness - self.n_games // 3  # 200 is arbitrary --> //3 means we explore much more time with constant randomness probability!!
        final_action = [0, 0, 0, 0, 0, 0]
        if random.randint(0, 200) < self.epsilon:  # 200 is arbitrary
            action = random.randint(0, 5)
            final_action[action] = 1
        else:
            state = np.array(state)
            current_state = torch.tensor(state, dtype=torch.float).to(self.device)
            prediction = self.brain(current_state)
            action = torch.argmax(prediction).item()
            final_action[action] = 1

        end = time.time()
        # print("get_action: ", end - start)

        return final_action

    def remember(self, state, action, reward, next_state, gameover):
        self.memory.append((state, action, reward, next_state, gameover))
        # append to a certain file
        with open("./alpha_3/memory/" + self.name + ".txt", "a") as f:
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
        # print("train_short_memory: ", end - start)

    def train_long_memory(self):
        start = time.time()
        if len(self.memory) > self.batch_size:
            batch_sample = random.sample(self.memory, self.batch_size)
            ''' Loss of the exploration phase in the long run
            batch_sample = []
            for i in range(len(self.memory)-1, len(self.memory) - 1 - BATCH_SIZE, -1):
                batch_sample.append(self.memory[i])
            '''
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
        if self.name == "seeker":
            print(f"\033[94mtraning seeker's long memory took: {end - start} seconds\033[0m")
        else:
            print(f"\033[92mtraning hider's long memory took: {end - start} seconds\033[0m")

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
        print("+" * 50)
        print(f"\033[96mtrain_replay with {criterion} criterion in: {end - start} seconds\033[0m")
        print("+" * 50)
        self.replay_memory = []

    def clean_memory(self, duplicates=100):
        start = time.time()
        # clean identical lines if number is over
        file_path = "./alpha_3/memory/" + self.name + ".txt"
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
        print(f"\033[92mclean_memory for {self.name}, erased {erased} lines in: ", end - start,
              " seconds\033[0m")
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
        start = time.time()
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

        end = time.time()
        # print("get_state: ", end - start)

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


    def train(self):
        """if len(self.memory) == 0 : return
        if len(self.memory) > self.batch_size:
            batch_sample = random.sample(self.memory, self.batch_size)
        else:
            batch_sample = self.memory"""
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

        # normalize everything
        #i = i / game.rows
        #j = j / game.cols
        distance = distance / (game.rows + game.cols)
        #other_player_i = other_player_i / game.rows
        #other_player_j = other_player_j / game.cols

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


    def train(self):
        """if len(self.memory) == 0 : return
        if len(self.memory) > self.batch_size:
            batch_sample = random.sample(self.memory, self.batch_size)
        else:
            batch_sample = self.memory"""
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


    def train(self):
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


    def train(self):
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