import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size_2, output_size, name='model'):
        super().__init__()

        self.name = name

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size_2)
        self.linear3 = nn.Linear(hidden_size_2, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def save(self):
        model_folder_path = '.\model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        path_name = os.path.join(model_folder_path, self.name)
        torch.save(self.state_dict(), path_name)

    def load(self):
        model_folder_path = '.\model'
        # check if model folder exists
        if not os.path.exists(model_folder_path):
            print("No model in path {}".format(model_folder_path))
            return
        path_name = os.path.join(model_folder_path, self.name)
        self.load_state_dict(torch.load(path_name))



class QNet(nn.Module):
    def __init__(self, layers, name='model'):
        super().__init__()

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
        model_folder_path = '.\model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        path_name = os.path.join(model_folder_path, self.name)
        torch.save(self.state_dict(), path_name)

    def load(self):
        model_folder_path = '.\model'
        # check if model folder exists
        if not os.path.exists(model_folder_path):
            print("No model in path {}".format(model_folder_path))
            return
        path_name = os.path.join(model_folder_path, self.name)
        self.load_state_dict(torch.load(path_name))




class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

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


