# Hide and Seek in a grid!
import pygame as pg
import numpy as np
from game import Game
from agent import Agent_alpha, Agent_hivemind
from models import QTrainer, QTrainer_beta_1
import sys
import argparse
import os
from util import plot_learning_curve, write_config

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

MAX_TIME = 1000
WINTIME = 4


def main():
    # Parse arguments to select game/training modes, map and Rendering options
    parser = argparse.ArgumentParser(description='Hide and Seek RL in a grid!')
    parser.add_argument('--hide', action='store_true', help='train the hider')
    parser.add_argument('--seek', action='store_true', help='train the seeker')
    parser.add_argument('--map', type=str, help='map name')
    parser.add_argument('--lidar', action='store_true', help='show lidar vision - can also be done pressing l in game')
    parser.add_argument('--view', action='store_true', help='show agent view - can also be done pressing v in game')
    parser.add_argument('--scores', action='store_true', help='show scores - can also be done pressing s in game')
    parser.add_argument('--random_spawn', action='store_true', help='spawn players randomly')
    args = parser.parse_args()

    hide = args.hide        # Train the hider or not
    seek = args.seek        # Train the seeker or not

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

    # Grid parameters
    agents = [Agent_alpha, Agent_hivemind]
    alphas = [0, 1, 2, 3]
    trainers = [QTrainer, QTrainer_beta_1]
    LRs = [0.1, 0.01, 0.001]
    batch_sizes = [1000, 5000, 10000]
    max_memories = [5000, 10000, 50000]
    seeker_reward_criterion = 'explore'
    hider_reward_criterion = 'explore'
    reward_replay_interval = 20
    neg_reward_replay_interval = 10
    clean_memory_interval = 30
    MAX_GENERATION = 500

    for ag in range(len(agents)):
        for alpha in range(len(alphas)):
            # If Hivemind has completed all hyperparameters configurations -> break
            if ag == 1 and alpha > 0: break
            for t in range(len(trainers)):
                for l in range(len(LRs)):
                    for b in range(len(batch_sizes)):
                        for m in range(len(max_memories)):

                            # Init Agents

                            # Agent_alpha
                            if ag == 0:
                                hider = agents[ag](alphas[alpha], 'hider_'+str(m)+str(b)+str(l)+str(t), trainers[t], LRs[l], batch_sizes[b], max_memories[m])
                                seeker = agents[ag](alphas[alpha], 'seeker_'+str(m)+str(b)+str(l)+str(t), trainers[t], LRs[l], batch_sizes[b], max_memories[m])
                            # Agent_hivemind
                            else:
                                hider = agents[ag]('hider_'+str(m)+str(b)+str(l)+str(t), trainers[t], LRs[l], batch_sizes[b], max_memories[m])
                                seeker = agents[ag]('seeker_'+str(m)+str(b)+str(l)+str(t), trainers[t], LRs[l], batch_sizes[b], max_memories[m])

                            if hide:
                                # Hider settings
                                hider_trainer = ""
                                if hider.Qtrainer == QTrainer:
                                    hider_trainer = "Qtrainer"
                                elif hider.Qtrainer == QTrainer_beta_1:
                                    hider_trainer = "QTrainer_beta_1"
                                hider_eps_dec = "-1 per game" 
                                hider_eps_min = "0"
                                hider_layers = hider.brain.conv_mlp_layers if hider.agent_name == "hivemind" else  hider.brain.layer_list
                                write_config(hider.agent_name, hider.name, map_name, hider_trainer, hider.lr, hider.batch_size, hider.max_memory, hider.epsilon, hider_eps_dec, hider_eps_min, hider_layers, hider_reward_criterion)
                            
                            if seek:
                                # Seeker settings
                                seeker_trainer = ""
                                if seeker.Qtrainer == QTrainer:
                                    seeker_trainer = "Qtrainer"
                                elif seeker.Qtrainer == QTrainer_beta_1:
                                    seeker_trainer = "QTrainer_beta_1"
                                seeker_eps_dec = "-1 per game" 
                                seeker_eps_min = "0"
                                seeker_layers = hider.brain.conv_mlp_layers if seeker.agent_name == "hivemind" else  seeker.brain.layer_list
                                write_config(seeker.agent_name, seeker.name, map_name, seeker_trainer, seeker.lr, seeker.batch_size, seeker.max_memory, seeker.epsilon, seeker_eps_dec, seeker_eps_min, seeker_layers, seeker_reward_criterion)

                            seeker_rewards, seeker_eps_history = [], []
                            hider_rewards, hider_eps_history = [], []


                            frames = 0
                            framerate = 60
                            gameover = False
                            stop = False
                            render = True

                            # GAME LOOP
                            while True:
                                # Close the window
                                for event in pg.event.get():
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
                                    game.players[0].look()
                                    game.players[0].trigger_lidar()
                                    hider_state = hider.get_state(game, game.players[0])
                                    hider_action = hider.get_action(hider_state)
                                    valid_action = game.control_player(game.players[0], hider_action)
                                    hider_reward = game.reward(game.players[0], valid_action, WINTIME, frames, MAX_TIME, criterion=hider_reward_criterion)
                                    hider_new_state = hider.get_state(game, game.players[0])
                                    hider.train_short_memory(hider_state, hider_action, hider_reward, hider_new_state, gameover)
                                    hider.remember(hider_state, hider_action, hider_reward, hider_new_state, gameover)
                                
                                # Seeker turn
                                if not frames % 2 and seek:
                                    game.players[1].look()
                                    game.players[1].trigger_lidar()
                                    seeker_state = seeker.get_state(game, game.players[1])
                                    seeker_action = seeker.get_action(seeker_state)
                                    valid_action = game.control_player(game.players[1], seeker_action)
                                    seeker_reward = game.reward(game.players[1], valid_action, WINTIME, frames, MAX_TIME, criterion=seeker_reward_criterion)
                                    seeker_new_state = seeker.get_state(game, game.players[1])
                                    seeker.train_short_memory(seeker_state, seeker_action, seeker_reward, seeker_new_state, gameover)
                                    seeker.remember(seeker_state, seeker_action, seeker_reward, seeker_new_state, gameover)


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

                                    # considering turn based operation, reward is only given to the agent that made the last move
                                    if frames % 2 and hide:
                                        hider_reward = game.reward(game.players[0], valid_action, WINTIME, frames, MAX_TIME, criterion=hider_reward_criterion)
                                        hider_state = hider.get_state(game, game.players[0])
                                        hider.remember(hider_state, [0 for i in range(hider.brain.layer_list[-1])], hider_reward, hider_state, gameover or stop)

                                    if not frames % 2 and seek:
                                        seeker_reward = game.reward(game.players[1], valid_action, WINTIME, frames, MAX_TIME, criterion=seeker_reward_criterion)
                                        seeker_state = seeker.get_state(game, game.players[1])
                                        seeker.remember(seeker_state, [0 for i in range(seeker.brain.layer_list[-1])], seeker_reward, seeker_state, gameover or stop)


                                    # Generation summary
                                    print("*" * 50)
                                    print(f"Generation: {seeker.n_games}")
                                    print("Exploring and Exploiting with:")
                                    if hide:
                                        print(f"\033[92mHider Epsilon: {hider.epsilon}\033[0m")
                                        print(f"\033[92mHider Reward: {game.players[0].reward}\033[0m")
                                    if seek: 
                                        print(f"\033[94mSeeker Epsilon: {seeker.epsilon}\033[0m")
                                        print(f"\033[94mSeeker Reward: {game.players[1].reward}\033[0m")

                                    if hide:
                                        hider_file_path = "./"+hider.agent_name+"/reward/reward_"+hider.name+".txt"
                                        if not os.path.exists("./"+hider.agent_name+"/reward"):
                                            os.makedirs("./"+hider.agent_name+"/reward")
                                        with open(hider_file_path, "a") as f:
                                            f.write(str(game.players[0].reward) + ";")

                                        hider.train_long_memory()

                                    if seek:
                                        seeker_file_path = "./"+seeker.agent_name+"/reward/reward_"+seeker.name+".txt"
                                        if not os.path.exists("./"+seeker.agent_name+"/reward"):
                                            os.makedirs("./"+seeker.agent_name+"/reward")
                                        with open(seeker_file_path, "a") as f:
                                            f.write(str(game.players[1].reward) + ";")

                                        seeker.train_long_memory()


                                    hider.n_games += 1
                                    seeker.n_games += 1

                                    if seek:
                                        seeker_rewards.append(game.players[1].reward)
                                        seeker_eps_history.append(seeker.epsilon)
                                        seeker_avg_reward = np.mean(seeker_rewards[-100:])
                                        print('Seeker average reward %.2f' % seeker_avg_reward)

                                        seeker_filename = 'seeker_'+seeker.agent_name+'_'+str(m)+str(b)+str(l)+str(t)+'.png'
                                        if seeker.n_games % 10 == 0 and seeker.n_games != 0:
                                            x = [i + 1 for i in range(seeker.n_games)]
                                            plot_learning_curve(x, seeker_rewards, seeker_eps_history, seeker_filename, seeker.agent_name, 'Seeker')

                                    if hide:
                                        hider_rewards.append(game.players[0].reward)
                                        hider_eps_history.append(hider.epsilon)
                                        hider_avg_reward = np.mean(hider_rewards[-100:])
                                        print('Hider average reward %.2f' % hider_avg_reward)

                                        hider_filename = 'hider_'+hider.agent_name+'_'+str(m)+str(b)+str(l)+str(t)+'.png'
                                        if hider.n_games % 10 == 0 and hider.n_games != 0:
                                            x = [i + 1 for i in range(hider.n_games)]
                                            plot_learning_curve(x, hider_rewards, hider_eps_history, hider_filename, hider.agent_name, 'Hider')

                                    
                                    # Train using replay memory and clean memory file every arbitrary number of games
                                    if (hider.n_games % reward_replay_interval == 0 or seeker.n_games % reward_replay_interval == 0) and (hider.n_games != 0 and seeker.n_games != 0):
                                        if hide : hider.train_replay("reward")
                                        if seek : seeker.train_replay("reward")
                                    
                                    if (hider.n_games % neg_reward_replay_interval == 0 or seeker.n_games % neg_reward_replay_interval == 0) and (hider.n_games != 0 and seeker.n_games != 0):
                                        if hide : hider.train_replay("neg_reward")
                                        if seek : seeker.train_replay("neg_reward")
                                    
                                    if (hider.n_games % clean_memory_interval == 0 or seeker.n_games % clean_memory_interval == 0) and (hider.n_games != 0 and seeker.n_games != 0):
                                        if hide : hider.clean_memory(duplicates=5)
                                        if seek : seeker.clean_memory(duplicates=5)


                                    # Reset game
                                    game.reset()
                                    frames = 0
                                    gameover = False
                                    stop = False
                                    
                                    # Break if generation limit is reached
                                    if (hider.n_games == MAX_GENERATION or seeker.n_games == MAX_GENERATION): break

            

if __name__ == "__main__":
    main()