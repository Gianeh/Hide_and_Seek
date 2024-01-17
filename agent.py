import torch
import random
from collections import deque
import numpy as np
from models import QTrainer, QNet, ConvQNet
import os

# import time for probing purposes
import time

MAX_MEMORY = 100000
BATCH_SIZE = 10000
LR = 0.03

# Agent Alpha 0 is the first complete prototype, every piece work but learning seems capped by input encoding or other unknown factors
class Agent_alpha_0:
    def __init__(self, name='model'):
        self.name = name
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.randomness = 200
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # automatic popleft()
        self.init_memory() # reload all previous memeories up to MAX_MEMORY
        self.replay_memory = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.brain = QNet([79, 256, 512, 256, 5], self.name).to(self.device)
        self.trainer = QTrainer(self.brain, LR, self.gamma)

        if self.brain.load():
            print("Model loaded")
        
    def init_memory(self):
        # check if memory file exists
        if not os.path.exists("./memory/" + self.name +".txt"):
            return
        # recall last lines of memory up to MAX_MEMORY
        with open("./memory/" + self.name +".txt", "r") as f:
            lines = f.readlines()
            if len(lines) > MAX_MEMORY:
                lines = lines[-MAX_MEMORY:]
            for line in lines:
                state, action, reward, next_state, gameover = line.split(";")
                state = np.array(state[1:-1].split(","), dtype=np.float32)
                action = np.array(action[1:-1].split(","), dtype=np.float32)
                reward = np.array(int(reward))
                next_state = np.array(next_state[1:-1].split(","), dtype=np.float32)
                gameover = np.array(gameover == 'True')
                self.memory.append((state, action, reward, next_state, gameover))

    def load_replay_memory(self, criterion="reward", size=BATCH_SIZE):
        with open("./memory/" + self.name +".txt", "r") as f:
            if criterion == "abs_reward":
                crit = lambda x: abs(int(x.split(";")[2]))
                reverse = True
            elif criterion == "reward":
                crit = lambda x: int(x.split(";")[2])
                reverse = True
            elif criterion == "neg_reward":
                crit = lambda x: int(x.split(";")[2])
                reverse = False
            elif criterion == "lowest_abs_reward":
                crit = lambda x: abs(int(x.split(";")[2]))
                reverse = False

            lines = sorted(f.readlines(), key=crit, reverse=reverse)

        if len(lines) > size:
            lines = lines[:size]
        for line in lines:
            state, action, reward, next_state, gameover = line.split(";")
            state = np.array(state[1:-1].split(","), dtype=np.float32)
            action = np.array(action[1:-1].split(","), dtype=np.float32)
            reward = np.array(int(reward))
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
        with open("./memory/" + self.name +".txt", "a") as f:
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
        if len(self.memory) > BATCH_SIZE:
            batch_sample = random.sample(self.memory, BATCH_SIZE)
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

    def train_replay(self, criterion="reward", size=BATCH_SIZE):
        start = time.time()

        self.load_replay_memory(criterion, size)

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
        file_path = "./memory/" + self.name + ".txt"
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
    def __init__(self, name='model'):
        self.name = name
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.randomness = 80
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # automatic popleft()
        self.init_memory() # reload all previous memeories up to MAX_MEMORY
        self.replay_memory = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.brain = QNet([19, 128, 128, 6], self.name).to(self.device)
        self.trainer = QTrainer(self.brain, LR, self.gamma)

        if self.brain.load():
            print("Model loaded")

        print(f"AGENT ALPHA 1: training {self.name} with {self.device} device")
        
    def init_memory(self):
        # check if memory file exists
        if not os.path.exists("./memory/" + self.name +".txt"):
            return
        # recall last lines of memory up to MAX_MEMORY
        with open("./memory/" + self.name +".txt", "r") as f:
            lines = f.readlines()
            if len(lines) > MAX_MEMORY:
                lines = lines[-MAX_MEMORY:]
            for line in lines:
                state, action, reward, next_state, gameover = line.split(";")
                state = np.array(state[1:-1].split(","), dtype=np.float32)
                action = np.array(action[1:-1].split(","), dtype=np.float32)
                reward = np.array(int(reward))
                next_state = np.array(next_state[1:-1].split(","), dtype=np.float32)
                gameover = np.array(gameover == 'True')
                self.memory.append((state, action, reward, next_state, gameover))

    def load_replay_memory(self, criterion="reward", size=BATCH_SIZE):
        with open("./memory/" + self.name +".txt", "r") as f:
            if criterion == "abs_reward":
                crit = lambda x: abs(int(x.split(";")[2]))
                reverse = True
            elif criterion == "reward":
                crit = lambda x: int(x.split(";")[2])
                reverse = True
            elif criterion == "neg_reward":
                crit = lambda x: int(x.split(";")[2])
                reverse = False
            elif criterion == "lowest_abs_reward":
                crit = lambda x: abs(int(x.split(";")[2]))
                reverse = False

            lines = sorted(f.readlines(), key=crit, reverse=reverse)

        if len(lines) > size:
            lines = lines[:size]
        for line in lines:
            state, action, reward, next_state, gameover = line.split(";")
            state = np.array(state[1:-1].split(","), dtype=np.float32)
            action = np.array(action[1:-1].split(","), dtype=np.float32)
            reward = np.array(int(reward))
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
        with open("./memory/" + self.name +".txt", "a") as f:
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
        if len(self.memory) > BATCH_SIZE:
            batch_sample = random.sample(self.memory, BATCH_SIZE)
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

    def train_replay(self, criterion="reward", size=BATCH_SIZE):
        start = time.time()

        self.load_replay_memory(criterion, size)

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
        file_path = "./memory/" + self.name + ".txt"
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
    

class Agent_hivemind_0:
    def __init__(self, name='model'):
        self.name = name
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.randomness = 200
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # automatic popleft()
        self.init_memory() # reload all previous memories up to MAX_MEMORY
        self.replay_memory = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.brain = ConvQNet([[1, 3, 3, 1, 1], [3, 1, 7, 1, 0]], [36, 128, 6], self.name).to(self.device)
        self.trainer = QTrainer(self.brain, LR, self.gamma, convolutional=True)

        if self.brain.load():
            print("Model loaded")

        print(f"AGENT HIVEMIND 0: training {self.name} with {self.device} device")
        
    def init_memory(self):
        # check if memory file exists
        if not os.path.exists("./memory/" + self.name +".txt"):
            return
        # recall last lines of memory up to MAX_MEMORY
        with open("./memory/" + self.name +".txt", "r") as f:
            lines = f.readlines()
            if len(lines) > MAX_MEMORY:
                lines = lines[-MAX_MEMORY:]
            for line in lines:
                state, action, reward, next_state, gameover = line.split(";")
                state = np.array(state[1:-1].split(","), dtype=np.float32)
                action = np.array(action[1:-1].split(","), dtype=np.float32)
                reward = np.array(int(reward))
                next_state = np.array(next_state[1:-1].split(","), dtype=np.float32)
                gameover = np.array(gameover == 'True')
                self.memory.append((state, action, reward, next_state, gameover))

    def load_replay_memory(self, criterion="reward", size=BATCH_SIZE):
        with open("./memory/" + self.name +".txt", "r") as f:
            if criterion == "abs_reward":
                crit = lambda x: abs(int(x.split(";")[2]))
                reverse = True
            elif criterion == "reward":
                crit = lambda x: int(x.split(";")[2])
                reverse = True
            elif criterion == "neg_reward":
                crit = lambda x: int(x.split(";")[2])
                reverse = False
            elif criterion == "lowest_abs_reward":
                crit = lambda x: abs(int(x.split(";")[2]))
                reverse = False

            lines = sorted(f.readlines(), key=crit, reverse=reverse)

        if len(lines) > size:
            lines = lines[:size]
        for line in lines:
            state, action, reward, next_state, gameover = line.split(";")
            state = np.array(state[1:-1].split(","), dtype=np.float32)
            action = np.array(action[1:-1].split(","), dtype=np.float32)
            reward = np.array(int(reward))
            next_state = np.array(next_state[1:-1].split(","), dtype=np.float32)
            gameover = np.array(gameover == 'True')
            self.replay_memory.append((state, action, reward, next_state, gameover))
                
    # hivemind agent is designed to acquire the matrix of the whole map
    def get_state(self, game, player):
        start = time.time()
        objects = {'wall': '5', 'floor': '4', 'hider': '3', 'movable_wall': '2','seeker': '1'}
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
        self.epsilon = self.randomness - self.n_games    # 80 is arbitrary
        final_action = [0,0,0,0,0,0]
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
        '''
        with open("./memory/" + self.name +".txt", "a") as f:
            f.write(str(state) + ";")
            f.write(str(action) + ";")
            f.write(str(reward) + ";")
            f.write(str(next_state) + ";")
            f.write(str(gameover) + "\n")
        '''

    def train_short_memory(self, state, action, reward, next_state, gameover):
        start = time.time()
        state = np.array(state)
        next_state = np.array(next_state)
        self.trainer.train_step(state, action, reward, next_state, gameover)
        end = time.time()
        #print("train_short_memory: ", end - start)

    def train_long_memory(self):
        start = time.time()
        if len(self.memory) > BATCH_SIZE:
            batch_sample = random.sample(self.memory, BATCH_SIZE)
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

    def train_replay(self, criterion="reward", size=BATCH_SIZE):
        start = time.time()

        self.load_replay_memory(criterion, size)

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
        file_path = "./memory/" + self.name + ".txt"
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
