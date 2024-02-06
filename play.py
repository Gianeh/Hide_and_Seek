import numpy as np
import pygame as pg
from game import Game
from agent import Agent_alpha, Agent_hivemind, Agent_beta
from models import QTrainer_beta_1, QTrainer
import sys
import argparse
import os
from util import plot_learning_curve, write_config

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

MAX_TIME = 1000
WINTIME = 4


def main():
    # parse arguments to select which agent to train
    parser = argparse.ArgumentParser(description='Hide and Seek RL in a grid!')
    parser.add_argument('--hide', action='store_true', help='train the hider')
    parser.add_argument('--seek', action='store_true', help='train the seeker')
    parser.add_argument('--map', type=str, help='map name')
    parser.add_argument('--lidar', action='store_true', help='show lidar vision - can also be done pressing l in game')
    parser.add_argument('--view', action='store_true', help='show agent view - can also be done pressing v in game')
    parser.add_argument('--scores', action='store_true', help='show scores - can also be done pressing s in game')
    parser.add_argument('--random_spawn', action='store_true', help='spawn players randomly')
    args = parser.parse_args()

    hide = args.hide
    seek = args.seek

    lidar = args.lidar
    view = args.view
    scores = args.scores
    random_spawn = args.random_spawn

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

    # Init Agents

    # Beta Agents
    hider = Agent_beta(beta=4, name='hider')
    hider.epsilon = 0.1
    seeker = Agent_beta(beta=4, name='seeker') 
    seeker.epsilon =0.1
    
    '''
    # Alpha Agents
    hider = Agent_alpha(alpha=3, name='hider')
    hider.epsilon = 10
    seeker = Agent_alpha(alpha=3, name='seeker')
    seeker.epsilon = 10
    '''
    '''
    # Hivemind Agents
    hider = Agent_hivemind(name='hider')
    hider.epsilon = 10
    seeker = Agent_hivemind(name='seeker')
    seeker.epsilon = 10
    '''

    hider_reward_criterion = 'smart_evasion'
    
    seeker_reward_criterion = 'smart_evasion'

    seeker_rewards, seeker_eps_history = [], []
    hider_rewards, hider_eps_history = [], []

    frames = 0
    framerate = 60
    gameover = False
    stop = False
    render = True

    while True:

        for event in pg.event.get():
            # close the window
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()

            # in-game commands
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    pg.quit()
                    sys.exit()
                if event.key == pg.K_SPACE:
                    # toggle graphics rendering and limited framerate
                    render = not render
                if event.key == pg.K_e:
                    # explore
                    hider.epsilon = 1.0
                    seeker.epsilon = 1.0
                    print("Agents can now explore again!")
                if event.key == pg.K_x:
                    # exploit
                    hider.epsilon = hider.eps_min
                    seeker.epsilon = seeker.eps_min
                    print(f"Agents are now exploiting with epsilon -> hider: {hider.eps_min} seeker: {seeker.eps_min}!")
                if event.key == pg.K_UP:
                    # increase framerate
                    framerate += 10
                    print(f"framerate up: {framerate}")
                if event.key == pg.K_DOWN:
                    # decrease framerate
                    framerate -= 10
                    print(f"framerate down: {framerate}")
                    if framerate < 0: framerate = 1
                if event.key == pg.K_p:
                    # pause
                    pg.event.wait()
                    print("Game paused!")
                if event.key == pg.K_l:
                    lidar = not lidar
                    print("Toggled Lidar visualization")
                if event.key == pg.K_v:
                    view = not view
                    print("Toggled Agent View visualization")
                if event.key == pg.K_s:
                    scores = not scores
                    print("Toggled Scores visualization")


        # let the Agents control the players

        if frames % 2 and hide:
            game.players[0].look()
            game.players[0].trigger_lidar()
            hider_state = hider.get_state(game, game.players[0])
            hider_action = hider.get_action(hider_state)
            valid_action = game.control_player(game.players[0], hider_action)
            hider_reward = game.reward(game.players[0], valid_action, WINTIME, frames, MAX_TIME, criterion=hider_reward_criterion)

        if not frames % 2 and seek:
            game.players[1].look()
            game.players[1].trigger_lidar()
            seeker_state = seeker.get_state(game, game.players[1])
            seeker_action = seeker.get_action(seeker_state)
            valid_action = game.control_player(game.players[1], seeker_action)
            seeker_reward = game.reward(game.players[1], valid_action, WINTIME, frames, MAX_TIME, criterion=seeker_reward_criterion)

        frames += 1

        # check if gameover
        if game.players[1].seen >= WINTIME:
            gameover = True

        if frames >= MAX_TIME:
            stop = True

        # update screen
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

            if not frames % 2 and seek:
                seeker_reward = game.reward(game.players[1], valid_action, WINTIME, frames, MAX_TIME, criterion=seeker_reward_criterion)



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

            hider.n_games += 1
            seeker.n_games += 1

            if seek:
                seeker_rewards.append(game.players[1].reward)
                seeker_eps_history.append(seeker.epsilon)

                seeker_filename = 'seeker_'+seeker.agent_name+'_play.png'
                if seeker.n_games % 10 == 0 and seeker.n_games != 0:
                    x = [i + 1 for i in range(seeker.n_games)]
                    plot_learning_curve(x, seeker_rewards, seeker_eps_history, seeker_filename, seeker.agent_name, 'Seeker', True)

            if hide:
                hider_rewards.append(game.players[0].reward)
                hider_eps_history.append(hider.epsilon)
  
                hider_filename = 'hider_'+hider.agent_name+'_play.png'
                if hider.n_games % 10 == 0 and hider.n_games != 0:
                    x = [i + 1 for i in range(hider.n_games)]
                    plot_learning_curve(x, hider_rewards, hider_eps_history, hider_filename, hider.agent_name, 'Hider', True)


            game.reset()
            frames = 0
            gameover = False
            stop = False


if __name__ == "__main__":
    main()