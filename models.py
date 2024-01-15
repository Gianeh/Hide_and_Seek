import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class QNet(nn.Module):
    def __init__(self, layers, name='model'):
        super().__init__()

        self.name = name
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))

    def forward(self, x):
        x
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        return x

    def save(self):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        path_name = os.path.join(model_folder_path, self.name)
        torch.save(self.state_dict(), path_name)

    def load(self):
        model_folder_path = './model'
        # check if model folder exists
        if not os.path.exists(model_folder_path):
            print("No model in path {}".format(model_folder_path))
            return False
        path_name = os.path.join(model_folder_path, self.name)
        self.load_state_dict(torch.load(path_name))
        return True




class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(self.device)
        action = torch.tensor(action, dtype=torch.long).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.device)
        # (n, x)

        # in case of short memory training - online training
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


        # state_0 -> pred_0 = [0.1, 0.3, 1.0, 0.2, 0.0] -> action_0 = [0, 0, 1, 0, 0] -> target_0 = [0.1, 0.3, Q_new, 0.2, 0.0]
        # state_new -> pred_new = [1.5, 0.1, 0.1, 0.1, 0.0] -> Q_new = reward_0 + gamma * max(pred_new)
        #                                                                                       ^^^^ (1.5)


