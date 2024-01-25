import numpy as np
import pygame as pg
from game import Game
from agent import Agent_alpha_6
from models import QTrainer_beta_1
import sys
import argparse
import os
from util import plot_learning_curve

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

MAX_TIME = 1000
WINTIME = 3


def main():
    # parse arguments to select which agent to train
    parser = argparse.ArgumentParser(description='Hide and Seek RL in a grid!')
    parser.add_argument('--hide', action='store_true', help='train the hider')
    parser.add_argument('--seek', action='store_true', help='train the seeker')
    args = parser.parse_args()

    hide = args.hide
    seek = args.seek

    # Instantiate Game and Agents
    game = Game(26, 25, 40)
    hider = Agent_alpha_6('hider', eps_min=0.1)
    seeker = Agent_alpha_6('seeker', eps_min=0.1)

    seeker_rewards, eps_history = [], []

    frames = 0
    framerate = 60
    gameover = False
    stop = False
    render = True

    while True:
        # close the window
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    pg.quit()
                    sys.exit()
                if event.key == pg.K_SPACE:
                    # toggle graphics rendering and limited framerate
                    render = not render
                if event.key == pg.K_e:
                    # explore
                    hider.randomness = 80
                    seeker.randomness = 80
                if event.key == pg.K_x:
                    # exploit
                    hider.randomness = 0
                    seeker.randomness = 0
                if event.key == pg.K_UP:
                    # increase framerate
                    framerate += 10
                if event.key == pg.K_DOWN:
                    # decrease framerate
                    framerate -= 10
                    if framerate < 0: framerate = 1
                if event.key == pg.K_p:
                    # pause
                    pg.event.wait()

        # let the Agents control the players

        if frames % 2 and hide:
            game.players[0].look()
            hider_state = hider.get_state(game, game.players[0])
            hider_action = hider.get_action(hider_state)
            valid_action = game.control_player(game.players[0], hider_action)
            hider_reward = game.reward(game.players[0], valid_action, WINTIME, criterion="explore")
            hider_new_state = hider.get_state(game, game.players[0])
            #hider.train()
            hider.remember(hider_state, hider_action, hider_reward, hider_new_state, gameover)

        if not frames % 2 and seek:
            game.players[1].look()
            seeker_state = seeker.get_state(game, game.players[1])
            seeker_action = seeker.get_action(seeker_state)
            valid_action = game.control_player(game.players[1], seeker_action)
            seeker_reward = game.reward(game.players[1], valid_action, WINTIME, criterion="explore")
            seeker_new_state = seeker.get_state(game, game.players[1])
            #seeker.train()
            seeker.remember(seeker_state, seeker_action, seeker_reward, seeker_new_state, gameover)

        frames += 1

        # update screen
        if render:
            game.screen.fill(WHITE)
            game.update()
            pg.display.flip()
            game.clock.tick(framerate)

        # check if gameover
        if game.players[1].seen >= WINTIME:
            gameover = True

        if frames >= MAX_TIME:
            stop = True

        if gameover or stop:

            # log some info about generation
            print("*" * 50)
            print(f"Generation: {seeker.n_games}")
            print(f"Exploring and Exploiting with Epsilon: {seeker.epsilon}")
            print(f"\033[94mSeeker Reward: {game.players[1].reward}\033[0m")
            print(f"\033[92mHider Reward: {game.players[0].reward}\033[0m")

            hider_file_path = "./" + hider.agent_name + "/reward/reward_" + hider.name + ".txt"
            if not os.path.exists("./" + hider.agent_name + "/reward"):
                os.makedirs("./" + hider.agent_name + "/reward")
            with open(hider_file_path, "a") as f:
                f.write(str(game.players[0].reward) + ";")

            seeker_file_path = "./" + seeker.agent_name + "/reward/reward_" + seeker.name + ".txt"
            if not os.path.exists("./" + seeker.agent_name + "/reward"):
                os.makedirs("./" + seeker.agent_name + "/reward")
            with open(seeker_file_path, "a") as f:
                f.write(str(game.players[1].reward) + ";")


            if hide: hider.train()
            if seek: seeker.train()

            hider.n_games += 1
            seeker.n_games += 1

            seeker_rewards.append(game.players[1].reward)
            eps_history.append(seeker.epsilon)
            avg_reward = np.mean(seeker_rewards[-100:])
            print('Game: ', seeker.n_games, ' Seeker reward %.2f' % game.players[1].reward, 'average reward %.2f' % avg_reward, 'epsilon %.2f' % seeker.epsilon)

            #filename = 'alpha6.png'
            filename = seeker.agent_name+'.png'
            if seeker.n_games % 50 == 0:
                x = [i + 1 for i in range(seeker.n_games)]
                plot_learning_curve(x, seeker_rewards, eps_history, filename)


            game.reset()
            frames = 0
            gameover = False
            stop = False


if __name__ == "__main__":
    main()