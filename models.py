import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

# Feedforward neural network (MLP)
class QNet(nn.Module):
    def __init__(self, layers, agent_name, name='model'):
        super().__init__()
        self.layer_list = layers        # Local variable to be saved in the config file
        self.agent_name = agent_name    # Alpha_0 or Alpha_1 or ...
        self.name = name                # Hider or Seeker
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        return x

    # Save Model to file in the agent directory
    def save(self):
        model_folder_path = os.path.join('./'+self.agent_name, 'model')
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        path_name = os.path.join(model_folder_path, self.name)
        torch.save(self.state_dict(), path_name)

    # Load Model from file in the agent directory
    def load(self):
        model_folder_path = os.path.join('./'+self.agent_name, 'model')
        # Check if model file in folder model exists
        path_name = os.path.join(model_folder_path, self.name)
        if not os.path.exists(path_name):
            print(f"No model in path {path_name}")
            return False
        self.load_state_dict(torch.load(path_name))
        return True

# Convolutional + MLP neural network
class ConvQNet(nn.Module): 
    # conv_layers = [[n_in_channels, n_out_channels, kernel_size, stride, padding], ...]
    def __init__(self, conv_layers, mlp_layers, agent_name, name='model'):
        super().__init__()
        self.agent_name = agent_name   # Alpha_0 or Alpha_1 or ...
        self.name = name                # Hider or Seeker
        self.layer_list = mlp_layers
        self.conv_mlp_layers = conv_layers + mlp_layers
        self.conv_layers = nn.ModuleList()
        for layer in conv_layers:
            self.conv_layers.append(nn.Conv2d(in_channels=layer[0], out_channels=layer[1], kernel_size=layer[2],
                                                stride=layer[3], padding=layer[4]))

        self.mlp_layers = nn.ModuleList()
        for i in range(len(mlp_layers) - 1):
            self.mlp_layers.append(nn.Linear(mlp_layers[i], mlp_layers[i + 1]))

    def forward(self, x):
        if len(x.shape) == 2:  # Only height and width present
            x = x.unsqueeze(0)  # Add channel dimension

        for layer in self.conv_layers:
            x = F.relu(layer(x))

        # flatten
        x = x.view(x.size(0), -1)
        
        for layer in self.mlp_layers[:-1]:
            x = F.relu(layer(x))
        x = self.mlp_layers[-1](x)
        return x
    
    # Save Model to file in the agent directory
    def save(self):
        model_folder_path = os.path.join('./'+self.agent_name, 'model')
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        path_name = os.path.join(model_folder_path, self.name)
        torch.save(self.state_dict(), path_name)

    # Load Model from file in the agent directory
    def load(self):
        model_folder_path = os.path.join('./'+self.agent_name, 'model')
        # Check if model file in folder model exists
        path_name = os.path.join(model_folder_path, self.name)
        if not os.path.exists(path_name):
            print("No model in path {}".format(path_name))
            return False
        self.load_state_dict(torch.load(path_name))
        return True

# Q-Learning Algortihm using Bellman Equation
class QTrainer:
    def __init__(self, model, lr, gamma, convolutional=False):
        self.lr = lr
        self.gamma = gamma  # Discount factor (takes into account future expected rewards)
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)

        # MSE loss is a standard for Q-learning
        self.criterion = nn.MSELoss()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Distingush between Convolutional and Feedforward models
        self.convolutional = convolutional

    # Train the model Online or in Batch mode
    def train_step(self, state, action, reward, next_state, done):

        # Set the model to train mode
        self.model.train()

        # Numpy arrays to tensors on the selected device
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(self.device)
        action = torch.tensor(action, dtype=torch.int32).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.device)

        # If using Q_Net
        if not self.convolutional:
            # In case of Online training -> Convert to batch mode compatible data (single element batch - Matrix)
            if len(state.shape) == 1:
                state = torch.unsqueeze(state, 0)
                next_state = torch.unsqueeze(next_state, 0)
                action = torch.unsqueeze(action, 0)
                reward = torch.unsqueeze(reward, 0)
                done = (done, )     # Done is a tuple just to get it's length in training loop

        # If using ConvQ_Net
        else:
            # In case of Online training -> Convert to batch mode compatible data (single element batch - 3d Tensor)
            if len(state.shape) == 2:
                state = torch.unsqueeze(state,0)
                next_state = torch.unsqueeze(next_state,0)
                action = torch.unsqueeze(action,0)
                reward = torch.unsqueeze(reward,0)
                done = (done, )     # Done is a tuple just to get it's length in training loop


            # Add channel dimension for single level input
            state = torch.unsqueeze(state, 1)
            next_state = torch.unsqueeze(next_state, 1)

        # 1: predicted Q values with current state
        pred = self.model(state)

        # Clone prediction to build the target tensor
        target = pred.detach().clone()

        # 2: Populate target for each step in the batch
        for idx in range(len(done)):
            Q_new = reward[idx]
            # Bellman Equation: Q_new = r + y * max(next_predicted Q value) -> only do this if not done (Game is not over)
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # 3: Backpropagation
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        # 4: Update weights
        self.optimizer.step()

        # Bellman equation example:
        # state_0 -> pred_0 = [0.1, 0.3, 1.0, 0.2, 0.0] -> action_0 = [0, 0, 1, 0, 0] -> target_0 = [0.1, 0.3, Q_new, 0.2, 0.0]
        # state_new -> pred_new = [1.5, 0.1, 0.1, 0.1, 0.0] -> Q_new = reward_0 + gamma * max(pred_new)
        #                                                                                       ^^^^ (1.5)

# Q-Learning Algortihm using Bellman Equation and a target network
class QTrainer_beta_1:
    def __init__(self, model, lr, gamma, convolutional=False, update_steps = 100):
        self.lr = lr
        self.gamma = gamma      # Discount factor (takes into account future expected rewards)
        self.model = model
        self.target_predictor = self.model  # Target network
        self.target_predictor.load_state_dict(self.model.state_dict())
        self.train_steps = 0
        self.update_steps = update_steps    # Update the target network every update_steps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)

        # MSE loss is a standard for Q-learning
        self.criterion = nn.MSELoss()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Distingush between Convolutional and Feedforward models
        self.convolutional = convolutional

    # Train the model Online or in Batch mode
    def train_step(self, state, action, reward, next_state, done):

        # Set the model to train mode
        self.model.train()

        # Count the number of training steps in order to update the target network after a while
        self.train_steps += 1
        
        # Update the target network
        if self.train_steps == self.update_steps:
            self.train_steps = 0
            # Copy the model parameters to the target network
            self.target_predictor.load_state_dict(self.model.state_dict())

        # Numpy arrays to tensors on the selected device
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(self.device)
        action = torch.tensor(action, dtype=torch.int).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.device)

        # If using Q_Net
        if not self.convolutional:
            # In case of Online training -> Convert to batch mode compatible data (single element batch - Matrix)
            if len(state.shape) == 1:
                state = torch.unsqueeze(state, 0)
                next_state = torch.unsqueeze(next_state, 0)
                action = torch.unsqueeze(action, 0)
                reward = torch.unsqueeze(reward, 0)
                done = (done,)  # Done is a tuple just to get it's length in training loop
        # If using ConvQ_Net
        else:
            # In case of Online training -> Convert to batch mode compatible data (single element batch - 3d Tensor)
            if len(state.shape) == 2:
                state = torch.unsqueeze(state, 0)
                next_state = torch.unsqueeze(next_state, 0)
                action = torch.unsqueeze(action, 0)
                reward = torch.unsqueeze(reward, 0)
                done = (done,)  # Done is a tuple just to get it's length in training loop

            # Add channel dimension for single level input
            state = torch.unsqueeze(state, 1)
            next_state = torch.unsqueeze(next_state, 1)

        # 1: predicted Q values with current state
        pred = self.model(state)

        # Clone prediction to build the target tensor
        target = pred.detach().clone()

        # 2: Populate target for each step in the batch
        for idx in range(len(done)):
            Q_new = reward[idx]
            # Bellman Equation: Q_new = r + y * max(next_predicted Q value) -> only do this if not done (Game is not over)
            if not done[idx]:
                # Target network is used to predict the next state (expected reward) instead of the model
                Q_new = reward[idx] + self.gamma * torch.max(self.target_predictor(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # 3: Backpropagation
        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()

        # 4: Update weights
        self.optimizer.step()