import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class QNet(nn.Module):
    def __init__(self, layers, agent_name, name='model'):
        super().__init__()

        self.agent_name = agent_name
        self.name = name
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        return x

    def save(self):
        model_folder_path = os.path.join('./'+self.agent_name, 'model')
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        path_name = os.path.join(model_folder_path, self.name)
        torch.save(self.state_dict(), path_name)

    def load(self):
        model_folder_path = os.path.join('./'+self.agent_name, 'model')
        # check if model file in folder model exists
        path_name = os.path.join(model_folder_path, self.name)
        if not os.path.exists(path_name):
            print("No model in path {}".format(path_name))
            return False
        self.load_state_dict(torch.load(path_name))
        return True

class ConvQNet(nn.Module):
    # conv_layers = [[n_in_channels, n_out_channels, kernel_size, stride, padding], [n_in_channels, n_out_channels, kernel_size, stride, padding], ...]
    def __init__(self, conv_layers, mlp_layers, agent_name, name='model'):
        super().__init__()
        self.agent_name = agent_name
        self.name = name
        self.conv_layers = nn.ModuleList()
        for layer in conv_layers:
            self.conv_layers.append(nn.Conv2d(in_channels=layer[0], out_channels=layer[1], kernel_size=layer[2],
                                                stride=layer[3], padding=layer[4]))
        # convolutional layers definition is not flexible and needs a coherent input

        self.mlp_layers = nn.ModuleList()
        for i in range(len(mlp_layers) - 1):
            self.mlp_layers.append(nn.Linear(mlp_layers[i], mlp_layers[i + 1]))

    def forward(self, x):
        if len(x.shape) == 2:  # Only height and width present
            x = x.unsqueeze(0)  # Add channel dimension

        for layer in self.conv_layers:
            x = F.leaky_relu(layer(x))

        # flatten
        x = x.view(x.size(0), -1)
        
        for layer in self.mlp_layers[:-1]:
            x = F.leaky_relu(layer(x))
        x = self.mlp_layers[-1](x)
        return x
    
    def save(self):
        model_folder_path = os.path.join('./'+self.agent_name, 'model')
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        path_name = os.path.join(model_folder_path, self.name)
        torch.save(self.state_dict(), path_name)

    def load(self):
        model_folder_path = os.path.join('./'+self.agent_name, 'model')
        # check if model file in folder model exists
        path_name = os.path.join(model_folder_path, self.name)
        if not os.path.exists(path_name):
            print("No model in path {}".format(path_name))
            return False
        self.load_state_dict(torch.load(path_name))
        return True


class QTrainer:
    def __init__(self, model, lr, gamma, convolutional=False):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.convolutional = convolutional

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(self.device)
        action = torch.tensor(action, dtype=torch.long).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.device)
        # (n, x)

        # in case of short memory training - online training
        if not self.convolutional:
            if len(state.shape) == 1:
                # (1, x)
                state = torch.unsqueeze(state, 0)
                next_state = torch.unsqueeze(next_state, 0)
                action = torch.unsqueeze(action, 0)
                reward = torch.unsqueeze(reward, 0)
                done = (done, )
        else:
            if len(state.shape) == 2:  # Only height and width present
                state = state.unsqueeze(0)  # Add channel dimension
                next_state = next_state.unsqueeze(0)
                done = (done, )
                action = action.unsqueeze(0)
                reward = reward.unsqueeze(0)


            # Add batch dimension
            state = torch.unsqueeze(state, 1)
            next_state = torch.unsqueeze(next_state, 1)

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()

        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()


        # state_0 -> pred_0 = [0.1, 0.3, 1.0, 0.2, 0.0] -> action_0 = [0, 0, 1, 0, 0] -> target_0 = [0.1, 0.3, Q_new, 0.2, 0.0]
        # state_new -> pred_new = [1.5, 0.1, 0.1, 0.1, 0.0] -> Q_new = reward_0 + gamma * max(pred_new)
        #                                                                                       ^^^^ (1.5)


class QTrainer_beta_1:
    def __init__(self, model, lr, gamma, convolutional=False, update_steps = 100):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.target_predictor = self.model
        self.target_predictor.load_state_dict(self.model.state_dict())
        self.train_steps = 0
        self.update_steps = update_steps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.convolutional = convolutional

    def train_step(self, state, action, reward, next_state, done):

        self.train_steps += 1

        if self.train_steps == self.update_steps:
            self.train_steps = 0
            self.target_predictor.load_state_dict(self.model.state_dict())


        state = torch.tensor(state, dtype=torch.float).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(self.device)
        action = torch.tensor(action, dtype=torch.long).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.device)
        # (n, x)

        # in case of short memory training - online training
        if not self.convolutional:
            if len(state.shape) == 1:
                # (1, x)
                state = torch.unsqueeze(state, 0)
                next_state = torch.unsqueeze(next_state, 0)
                action = torch.unsqueeze(action, 0)
                reward = torch.unsqueeze(reward, 0)
                done = (done,)
        else:
            if len(state.shape) == 2:  # Only height and width present
                state = state.unsqueeze(0)  # Add channel dimension
                next_state = next_state.unsqueeze(0)
                done = (done,)
                action = action.unsqueeze(0)
                reward = reward.unsqueeze(0)

            # Add batch dimension
            state = torch.unsqueeze(state, 1)
            next_state = torch.unsqueeze(next_state, 1)

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.detach().clone()

        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.target_predictor(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()

        self.optimizer.step()