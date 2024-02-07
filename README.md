# HIDE AND SEEK GAME


## INSTALLATION & USAGE:

### Install the requirements using the following command:
  
```pip install -r requirements.txt```

### Executions modes:
  Launch the training of the Alpha agents:
      
  ```python3 alpha_main.py```
  
  Launch the training of the Beta agents:
      
  ```python3 beta_main.py```
  
  Let the trained agents play against each other:
      
  ```python3 play.py```

### Execution arguments:
  
  --seek: let the seeker play (and train)
  
  --hide: let the hider play (and train)
  
  --perfect_seeker: let the perfect seeker play against the chosen agent for the hider - Doesn't work with the play.py script
  
  --map "map_name": choose the map to play on (empty: no walls or movable walls, movable: only movable walls, full: all types of walls)
  
  --random_spawn: let the agents spawn in random positions at every reset
  
  --view: render players' view
  
  --lidar: render players' lidar
  
  --score: render agents' score

### In-game commands:
  - ESC: close the game
  - SPACEBAR: stop the rendering to let the training continue at max speed
  - E: reset exploration rate to it's maximum (random actions chance)
  - X: let the agents play with minimum random actions chance (0 in alpha, eps_min in beta)
  - UP: increase the rendering framerate limit by 10 FPS
  - DOWN: decrease the rendering framerate limit by 10 FPS
  - L: toggle lidars' rendering
  - V: toggle players' view rendering
  - S: toggle agents' score rendering

### Configuration:
  - Agents can't be selected with arguments, they are imported in the main scripts and can be changed manually there
  - Alpha and Beta agents' generations are determined using the alpha and beta parameters in the class instantiation. alpha ranges from 0 to 3, beta from 0 to 4
  - Agent Hivemind is commented in the alpha_main.py script, it can be used to test the Conv_QNet model

### Loading pretrained agents:
  - Agents save their models at "./Agent_name/model"
  - Pretrained models can be loaded by adding the file in the respective folder

### Evaluation:
  - Plot all the saved rewards of the agents:
      
```python3 plot_rewards.py```

### Alpha agent grid search:
  - Launch the grid search of the alpha agents:

```python3 alpha_grid_search.py```
  - The grid search is configured in the alpha_grid_search.py script, it can be changed manually there



## OVERVIEW: 

### Game mechanics

The purpose of this project is to develop our version of the Hide and Seek game, where two players navigate custom maps with different levels of difficulty taking actions dictated by reinforcement learning algorithms.
The main goal was to observe some intelligent behaviours from the players during the training.
The game map is a matrix (26x26 blocks) composed of different types of object, like floor (walkable tile), movable wall (wall that can be grabbed/relased by players), wall (unmovable object), 
Hider (tries to hide from the seeker), Seeker (tries to catch the hider).
Each player has a view of the map, a list of map objects with depth of 4 blocks and width of 3 blocks used to determine the winning policy of the game as the seeker wins when,
for a certain amount of consecutive steps, his opponent is in view, while the hider wins if after a predefined amount of frames he isn't caught. 
The player's view is limited in depth by the first encountered obstacole that occlude his line of sight and by the map edges.

### Actions

The players, controlled by neural models' predictions, are capable of performing up to 6 actions, moving along the x and y directions, grabbing/releasing a movable_wall and standing still.

### Reinforcement learning

Multiple agents are developed and used to control the players' behaviours relying on the predictions of neural models based on the environment's state perception.
The models are trained through the the Deep Q-learning paradigm, using Bellman's equation to obtain a target value based on the current and future expected rewards.
The so called Q_values are used to perform gradient descent on the network weights to minimize the Mean Square Error with the previously taken action.
Generally speaking, the Q_values (Quality values of an action) are used to fill the Q_table of game states, in practice a cheatsheet of the best actions to take in a certain state.
Bellman's equation generates the target value for the Q_values, based on the current reward and the maximum expected future reward, discounted by a factor called gamma.

  ```Q_new = Q_(action_0, state_0) + gamma * max(Q_(action_1, state_1))```

Our Primary_QTrainer class takles the training process using the neural model to predict both Q_(action_0, state_0) and Q_(action_1, state_1) to adjust the weights of the model itself.
This may lead to instability, having the same network predicting the target and the Q_values, possibly resulting in a feedback loop.
To avoid this issue, Target_QTrainer class is built to use a copy of the neural model to predict the target (i.e. Q_(action_1, state_1)), updating only the weights of the original model.
The copy (Target predictor) is then repeated after a certain amount of steps, to keep the training process stable.


### Neural models

Policy networks, employed in the prediction of players' actions, are neural models of 2 different kinds:

  - QNet: a simple feedforward neural network with flexible number of inputs and layers architecture
  - Conv_QNet: a convolutional neural network with configurable number of layers, kernels, stride and padding, whose output is feeded to fully connected layers

  + Both models are trained using the Adam optimizer and the Mean Square Error loss function, The layers use ReLu activation function, except for the linear output layer


### Rewards

Rewards are a critical policy component in reinforcement learning, they provide the stimulus for the agents to learn the best actions to take, given a certain state.
A bad design, can lead to poor performances or unintended (lazy) behaviours, while a good one can be hard to define and a perfect one may not exist.
Our game employs 6 interchangable reward policies:

  - Default: every game step the seeker is punished and hider is rewarded if the first can't see the second
  - Explore: players aren't punished or rewarded is seeker can't see the hider, the latter is punished if he's seen
  - Explore2:  hider is rewarded if he's not seen but seeker is not punished and is rewarded if he sees the hider
  - Distance: hider and seeker rewards are respectively proportional to the distance and the inverse of the distance from the other player
  - Hybrid: a combination of Distance and Default criterions
  - Smart Evasion: hider only policy where the reward is based on the distance from the seeker depending on several factors such as a minimum and maximum value and a evasion success

  + Every criterion rewards the seeker if he catches the hider and punishes the hider if he's caught

### State representation

The state is what the neural model uses as input to determine the best action to take given the current environment's configuration. Our agents employ a combination of:

  - View: what the players see in the faced direction (3x4 matrix)
  - Position: player's position in the map, i,j indices or x,y coordinates (sometimes normalized)
  - Distance: distance from the other player (sometimes normalized)
  - Direction: player's faced direction (up, down, left, right)
  - Lidar: distance and object type of the nearest element along 8 axis vision system around the player
  - Neighbourhood: 3 blocks surrounding the player's position, excepting the faced one (inside the view)
  - Available actions: encoded list of valid moves the player can take
  - Entire map: encoded matrix of the entire map (26x26)

  + Objects' encoding vary from one-hot to scalar values, depending on the agent.


### Agents

The agents provide the interface between the neural models and the game environment, different states and model architectures are used depending on agent's generation:

  - Agent Alpha: collection of the first generation of agents using QNet, memory is saved to a file which provides the possibility to load it partially and/or selectively
  - Agent Hivemind: single agent experiment using Conv_QNet model, uses the encoded matrix of the entire map as state representation
  - Agent Beta: collection of the second generation of agents using QNet, memory is no longer saved to a file, avoiding replay memory training in advantage of the computational demand. Beta is the only agent that uses lidar vision
  - Agent Small Brain: in practice a Beta agent that uses a minimal state representation, composed of available actions and a vector of flags indicating respectively if the opponent is on his right, left, up or down sides
  - Perfect Seeker: a simple algorithm capable of findig the hider in the empty map. Used to test the hider's ability to escape



## REFERENCES:

Project's main inspiration comes from the following sources:

OpenAI Hide and Seek: https://openai.com/research/emergent-tool-use

Snake Reinforcement Learning in Pygame: https://www.youtube.com/watch?v=L8ypSXwyBds&ab_channel=freeCodeCamp.org

