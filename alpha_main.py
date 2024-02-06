import pygame as pg
import numpy as np
from game import Game
from agent import Agent_alpha, Agent_hivemind, Perfect_seeker
from models import Primary_QTrainer, Target_QTrainer
import sys
import argparse
import os
from util import plot_learning_curve, write_config

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

MAX_TIME = 1000
WINTIME = 4


def main():
    # Parse arguments to select game/training mode, map, and rendering options
    parser = argparse.ArgumentParser(description='Hide and Seek RL in a grid!')
    parser.add_argument('--hide', action='store_true', help='train the hider')
    parser.add_argument('--seek', action='store_true', help='train the seeker')
    parser.add_argument('--perfect_seeker', action='store_true', help='employ perfect seeker')
    parser.add_argument('--map', type=str, help='map name')
    parser.add_argument('--lidar', action='store_true', help='show lidar vision - can also be done pressing l in game')
    parser.add_argument('--view', action='store_true', help='show agent view - can also be done pressing v in game')
    parser.add_argument('--scores', action='store_true', help='show scores - can also be done pressing s in game')
    parser.add_argument('--random_spawn', action='store_true', help='spawn players randomly')
    args = parser.parse_args()

    hide = args.hide    # Train the hider or not
    seek = args.seek    # Train the seeker or not
    perfect_seeker = args.perfect_seeker    # Employ perfect seeker or not

    if perfect_seeker and seek:
        print("seeker can only be trained or perfect, can't be both! using perfect seeker")
        seek = False

    # Rendering options
    lidar = args.lidar
    view = args.view
    scores = args.scores

    # Random spawn option
    random_spawn = args.random_spawn

    # Map selection
    if args.map is None:
        print("No map chosen. using empty map!")
        map_name = 'empty.txt'
    elif args.map in "empty full movable":
        map_name = args.map + '.txt'
    else:
        print("Invalid map name. using empty map!")
        map_name = 'empty.txt'

    # Init Game
    game = Game(26, 26, 40, map_name, random_spawn)

    # Init Alpha Agents
    hider = Agent_alpha(alpha=3, name='hider', Qtrainer=Target_QTrainer, lr=0.001, batch_size=1000, max_memory=5000)
    seeker = Agent_alpha(alpha=3, name='seeker', Qtrainer=Target_QTrainer, lr=0.001, batch_size=1000, max_memory=5000) if not perfect_seeker else Perfect_seeker_0('seeker')
    
    '''
    # Init Hivemind Agents
    hider = Agent_hivemind(name='hider', Qtrainer=Target_QTrainer, lr=0.001, batch_size=1000, max_memory=5000)
    seeker = Agent_hivemind(name='seeker', Qtrainer=Target_QTrainer, lr=0.001, batch_size=1000, max_memory=5000) if not perfect_seeker else Perfect_seeker_0('seeker')
    '''

    if hide:
        # Hider settings
        hider_trainer = ""
        if hider.Qtrainer == Primary_QTrainer:
            hider_trainer = "Primary_QTrainer"
        elif hider.Qtrainer == Target_QTrainer:
            hider_trainer = "Target_QTrainer"
        hider_eps_dec = "-1 per game" 
        hider_eps_min = "0"
        hider_layers = hider.brain.conv_mlp_layers if hider.agent_name == "hivemind" else  hider.brain.layer_list

        hider_reward_criterion = 'explore'      # Reward criterion for the hider

        write_config(hider.agent_name, hider.name, map_name, hider_trainer, hider.lr, hider.batch_size, hider.max_memory, hider.epsilon, hider_eps_dec, hider_eps_min, hider_layers, hider_reward_criterion)
    
    if not perfect_seeker and seek:
        # Seeker settings
        seeker_trainer = ""
        if seeker.Qtrainer == Primary_QTrainer:
            seeker_trainer = "Primary_QTrainer"
        elif seeker.Qtrainer == Target_QTrainer:
            seeker_trainer = "Target_QTrainer"
        seeker_eps_dec = "-1 per game" 
        seeker_eps_min = "0"
        seeker_layers = hider.brain.conv_mlp_layers if seeker.agent_name == "hivemind" else  seeker.brain.layer_list

        seeker_reward_criterion = 'explore'     # Reward criterion for the seeker

        write_config(seeker.agent_name, seeker.name, map_name, seeker_trainer, seeker.lr, seeker.batch_size, seeker.max_memory, seeker.epsilon, seeker_eps_dec, seeker_eps_min, seeker_layers, seeker_reward_criterion)

    seeker_rewards, seeker_eps_history = [], []
    hider_rewards, hider_eps_history = [], []

    # Game render and logic parameters
    frames = 0
    framerate = 60
    gameover = False
    stop = False
    render = True

    # GAME LOOP
    while True:

        for event in pg.event.get():
            # Close the window
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()

            # In-game commands
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    pg.quit()
                    sys.exit()
                if event.key == pg.K_SPACE:
                    # Toggle graphics rendering and limited framerate
                    render = not render
                if event.key == pg.K_e:
                    # Explore
                    hider.epsilon = 200
                    seeker.epsilon = 200
                    print("Agents can now explore again!")
                if event.key == pg.K_x:
                    # Exploit
                    hider.epsilon = 0
                    seeker.epsilon = 0
                    print(f"Agents are now exploiting with epsilon -> hider: {hider.epsilon} seeker: {seeker.epsilon}!")
                if event.key == pg.K_UP:
                    # Increase framerate
                    framerate += 10
                    print(f"framerate up: {framerate}")
                if event.key == pg.K_DOWN:
                    # Decrease framerate
                    framerate -= 10
                    print(f"framerate down: {framerate}")
                    if framerate < 0: framerate = 1
                if event.key == pg.K_l:
                    # Render lidar vision
                    lidar = not lidar
                    print("Toggled Lidar visualization")
                if event.key == pg.K_v:
                    # Render agent view
                    view = not view
                    print("Toggled Agent View visualization")
                if event.key == pg.K_s:
                    # Render scores
                    scores = not scores
                    print("Toggled Scores visualization")


        # Let the Agents control the players
        # To avoid overlapping after both agents move, the game is turn based
                    
        # Hider turn
        if frames % 2 and hide:
            # Fill the view
            game.players[0].look()
            # Fill the lidar vision
            game.players[0].trigger_lidar()
            # Acquire current state
            hider_state = hider.get_state(game, game.players[0])
            # Get the next action
            hider_action = hider.get_action(hider_state)
            # Perform the action and make sure it was valid
            valid_action = game.control_player(game.players[0], hider_action)
            # Get the reward
            hider_reward = game.reward(game.players[0], valid_action, WINTIME, frames, MAX_TIME, criterion=hider_reward_criterion)
            # Get the new state
            hider_new_state = hider.get_state(game, game.players[0])
            # Online training
            hider.train_short_memory(hider_state, hider_action, hider_reward, hider_new_state, gameover)
            # Remember this game step
            hider.remember(hider_state, hider_action, hider_reward, hider_new_state, gameover)
        
        # Seeker turn
        if not frames % 2 and seek:
            # Fill the view
            game.players[1].look()
            # Fill the lidar vision
            game.players[1].trigger_lidar()
            # Acquire current state
            seeker_state = seeker.get_state(game, game.players[1])
            # Get the next action
            seeker_action = seeker.get_action(seeker_state)
            # Perform the action and make sure it was valid
            valid_action = game.control_player(game.players[1], seeker_action)
            # Get the reward
            seeker_reward = game.reward(game.players[1], valid_action, WINTIME, frames, MAX_TIME, criterion=seeker_reward_criterion)
            # Get the new state
            seeker_new_state = seeker.get_state(game, game.players[1])
            # Online training
            seeker.train_short_memory(seeker_state, seeker_action, seeker_reward, seeker_new_state, gameover)
            # Remember this game step
            seeker.remember(seeker_state, seeker_action, seeker_reward, seeker_new_state, gameover)

        # Perfect seeker turn
        if perfect_seeker and not frames % 2:
            # Fill the lidar vision
            game.players[1].trigger_lidar()
            # Acquire current state
            seeker_state = seeker.get_state(game, game.players[1])
            # Get the next action
            seeker_action = seeker.get_action(seeker_state)
            # Perform the action
            game.control_player(game.players[1], seeker_action)
            # Let seeker look and see in order to update the seen variable as it's not done in the reward function
            game.players[1].look()
            game.players[1].see()

        frames += 1

        # Check if seeker wins
        if game.players[1].seen >= WINTIME:
            gameover = True

        # Check if hider wins
        if frames >= MAX_TIME:
            stop = True

        # Update screen
        if render:

            if gameover or stop:
                game.players[0].look()
                game.players[1].look()

            game.screen.fill(WHITE)
            game.update(lidar = lidar, view = view, scores = scores)
            pg.display.flip()
            game.clock.tick(framerate)

        if gameover or stop:

            # Considering turn based operation, reward is only given to the agent that made the last move
            if frames % 2 and hide:
                hider_reward = game.reward(game.players[0], valid_action, WINTIME, frames, MAX_TIME, criterion=hider_reward_criterion)
                hider_state = hider.get_state(game, game.players[0])
                hider.remember(hider_state, [0 for i in range(hider.brain.layer_list[-1])], hider_reward, hider_state, gameover or stop)
                                            # Empty action

            if not frames % 2 and not perfect_seeker and seek:
                seeker_reward = game.reward(game.players[1], valid_action, WINTIME, frames, MAX_TIME, criterion=seeker_reward_criterion)
                seeker_state = seeker.get_state(game, game.players[1])
                seeker.remember(seeker_state, [0 for i in range(seeker.brain.layer_list[-1])], seeker_reward, seeker_state, gameover or stop)
                                            # Empty action


            # Generation summary
            print("*" * 50)
            print(f"Generation: {seeker.n_games}")
            print("Exploring and Exploiting with:")
            if hide:
                print(f"\033[92mHider Epsilon: {hider.epsilon}\033[0m")
                print(f"\033[92mHider Reward: {game.players[0].reward}\033[0m")
            if not perfect_seeker and seek: 
                print(f"\033[94mSeeker Epsilon: {seeker.epsilon}\033[0m")
                print(f"\033[94mSeeker Reward: {game.players[1].reward}\033[0m")

            
            # Save the epoch reward to a file
            if hide:
                hider_file_path = "./"+hider.agent_name+"/reward/reward_"+hider.name+".txt"
                if not os.path.exists("./"+hider.agent_name+"/reward"):
                    os.makedirs("./"+hider.agent_name+"/reward")
                with open(hider_file_path, "a") as f:
                    f.write(str(game.players[0].reward) + ";")

                # Batch training
                hider.train_long_memory()

            if seek:
                seeker_file_path = "./"+seeker.agent_name+"/reward/reward_"+seeker.name+".txt"
                if not os.path.exists("./"+seeker.agent_name+"/reward"):
                    os.makedirs("./"+seeker.agent_name+"/reward")
                with open(seeker_file_path, "a") as f:
                    f.write(str(game.players[1].reward) + ";")

                # Batch training
                seeker.train_long_memory()


            hider.n_games += 1
            seeker.n_games += 1


            # Log average reward and plot learning curve
            if not perfect_seeker and seek:
                seeker_rewards.append(game.players[1].reward)
                seeker_eps_history.append(seeker.epsilon)
                seeker_avg_reward = np.mean(seeker_rewards[-100:])
                print('Seeker average reward %.2f' % seeker_avg_reward)

                seeker_filename = 'seeker_'+seeker.agent_name+'.png'
                if seeker.n_games % 10 == 0 and seeker.n_games != 0:
                    x = [i + 1 for i in range(seeker.n_games)]
                    plot_learning_curve(x, seeker_rewards, seeker_eps_history, seeker_filename, seeker.agent_name, 'Seeker')

            if hide:
                hider_rewards.append(game.players[0].reward)
                hider_eps_history.append(hider.epsilon)
                hider_avg_reward = np.mean(hider_rewards[-100:])
                print('Hider average reward %.2f' % hider_avg_reward)

                hider_filename = 'hider_'+hider.agent_name+'.png'
                if hider.n_games % 10 == 0 and hider.n_games != 0:
                    x = [i + 1 for i in range(hider.n_games)]
                    plot_learning_curve(x, hider_rewards, hider_eps_history, hider_filename, hider.agent_name, 'Hider')

            
            # Train using replay memory and clean memory file every arbitrary number of games
            if (hider.n_games % 20 == 0 or seeker.n_games % 20 == 0) and (hider.n_games != 0 and seeker.n_games != 0):
                if hide : hider.train_replay("reward")
                if seek : seeker.train_replay("reward")
            
            if (hider.n_games % 10 == 0 or seeker.n_games % 10 == 0) and (hider.n_games != 0 and seeker.n_games != 0):
                if hide : hider.train_replay("neg_reward")
                if seek : seeker.train_replay("neg_reward")
            
            if (hider.n_games % 30 == 0 or seeker.n_games % 30 == 0) and (hider.n_games != 0 and seeker.n_games != 0):
                if hide : hider.clean_memory(duplicates=5)
                if seek : seeker.clean_memory(duplicates=5)

            # Reset the game
            game.reset()
            frames = 0
            gameover = False
            stop = False

            

if __name__ == "__main__":
    main()