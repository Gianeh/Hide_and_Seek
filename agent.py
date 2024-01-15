import torch
import random
from collections import deque
import numpy as np
from models import QTrainer, QNet

# import time for probing purposes
import time

MAX_MEMORY = 100000
BATCH_SIZE = 10000
LR = 0.01


class Agent:
    def __init__(self, name='model'):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # automatic popleft()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.brain = QNet([77, 256, 128, 5], name).to(self.device)
        self.trainer = QTrainer(self.brain, LR, self.gamma)

        if self.brain.load():
            self.n_games = 80 # avoid exploration on successive runs

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

        state = np.array([x_norm,y_norm] + view_vector + neighbourhood)

        end = time.time()
        #print("get_state: ", end - start)

        return state

    def get_action(self, state):
        start = time.time()
        # tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games    # 80 is arbitrary
        final_action = [0,0,0,0,0]
        if random.randint(0, 200) < self.epsilon:   # 200 is arbitrary
            action = random.randint(0, 4)
            final_action[action] = 1
        else:
            current_state = torch.tensor(state, dtype=torch.float).to(self.device)
            prediction = self.brain(current_state)
            action = torch.argmax(prediction).item()
            final_action[action] = 1
        
        end = time.time()
        #print("get_action: ", end - start)

        return final_action

    def remember(self, state, action, reward, next_state, gameover):
        self.memory.append((state, action, reward, next_state, gameover))

    def train_short_memory(self, state, action, reward, next_state, gameover):
        start = time.time()
        self.trainer.train_step(state, action, reward, next_state, gameover)
        end = time.time()
        #print("train_short_memory: ", end - start)

    def train_long_memory(self):
        start = time.time()
        if len(self.memory) > BATCH_SIZE:
            # batch_sample = random.sample(self.memory, BATCH_SIZE)
            batch_sample = []
            for i in range(len(self.memory)-1, len(self.memory) - 1 - BATCH_SIZE, -1):
                batch_sample.append(self.memory[i])
        else:
            batch_sample = self.memory
            print(type(batch_sample))

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
        print("train_long_memory: ", end - start)

    


