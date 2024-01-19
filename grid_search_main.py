import pygame as pg
from game import Game
from agent import Agent_alpha_0, Agent_alpha_1, Agent_alpha_2, Agent_hivemind_0
from models import QTrainer, QTrainer_beta_1
import sys
import argparse

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

MAX_TIME = 1000
WINTIME = 400


def main():
    # parse arguments to select which agent to train
    parser = argparse.ArgumentParser(description='Hide and Seek RL in a grid!')
    parser.add_argument('--hide', action='store_true', help='train the hider')
    parser.add_argument('--seek', action='store_true', help='train the seeker')
    args = parser.parse_args()

    hide = args.hide
    seek = args.seek

    # Instantiate Game and Agents
    game = Game(12, 12, 40)
    agents = [Agent_alpha_0, Agent_alpha_1, Agent_alpha_2, Agent_hivemind_0]
    trainers = [QTrainer, QTrainer_beta_1]
    LRs = [0.1, 0.05, 0.01, 0.005, 0.001]
    batch_sizes = [100, 500, 1000, 5000, 10000]
    max_memories = [500, 1000, 5000, 10000, 50000]
    MAX_GENERATION = 500

    #alpha_0 -> Done, alpha_1 -> Done, alpha_2 -> Done (need to be fixed -> stopping training at generation 50), hivemind_0 -> Done
    #for the moment Qtrainer_beta_1 use only default value for update_steps in his init method -> in future grid search update we'll need to pass this parameter to the selected agent



    for a in range(len(agents)):
        for t in range(len(trainers)):
            for l in range(len(LRs)):
                for b in range(len(batch_sizes)):
                    for m in range(len(max_memories)):
                        hider = agents[a]('hider_'+str(m)+str(b)+str(l)+str(t), trainers[t], LRs[l], batch_sizes[b], max_memories[m])
                        seeker = agents[a]('seeker'+str(m)+str(b)+str(l)+str(t), trainers[t], LRs[l], batch_sizes[b], max_memories[m])


                        frames = 0
                        framerate = 60
                        gameover = False
                        stop = False
                        render = False

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
                                hider.train_short_memory(hider_state, hider_action, hider_reward, hider_new_state, gameover)
                                hider.remember(hider_state, hider_action, hider_reward, hider_new_state, gameover)


                            if not frames % 2 and seek:
                                game.players[1].look()
                                seeker_state = seeker.get_state(game, game.players[1])
                                seeker_action = seeker.get_action(seeker_state)
                                valid_action = game.control_player(game.players[1], seeker_action)
                                seeker_reward = game.reward(game.players[1], valid_action, WINTIME, criterion="explore")
                                seeker_new_state = seeker.get_state(game, game.players[1])
                                seeker.train_short_memory(seeker_state, seeker_action, seeker_reward, seeker_new_state, gameover)
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
                                if seeker.epsilon > 0:
                                    print(f"Exploring and Exploiting with Epsilon: {seeker.epsilon}")
                                else:
                                    print("Exploiting only")

                                print(f"\033[92mHider Reward: {game.players[0].reward}\033[0m")
                                print(f"\033[94mSeeker Reward: {game.players[1].reward}\033[0m")

                                hider_file_path = "./"+hider.agent_name+"/reward_"+hider.name+".txt"
                                with open(hider_file_path, "a") as f:
                                    f.write(str(game.players[0].reward) + ";")

                                seeker_file_path = "./"+seeker.agent_name+"/reward_"+seeker.name+".txt"
                                with open(seeker_file_path, "a") as f:
                                    f.write(str(game.players[1].reward) + ";")

                                game.reset()
                                frames = 0
                                gameover = False
                                stop = False
                                hider.n_games += 1
                                if hide: hider.train_long_memory()
                                seeker.n_games += 1
                                if seek: seeker.train_long_memory()

                                """"
                                if (hider.n_games % 5 == 0 or seeker.n_games % 5 == 0) and (hider.n_games != 0 and seeker.n_games != 0):
                                    if hide: hider.train_replay("reward")
                                    if seek: seeker.train_replay("reward")

                                if (hider.n_games % 10 == 0 or seeker.n_games % 10 == 0) and (hider.n_games != 0 and seeker.n_games != 0):
                                    if hide: hider.train_replay("neg_reward")
                                    if seek: seeker.train_replay("neg_reward")
                                
                                if (hider.n_games % 30 == 0 or seeker.n_games % 30 == 0) and (hider.n_games != 0 and seeker.n_games != 0):
                                    if hide : hider.clean_memory(duplicates=5)
                                    if seek : seeker.clean_memory(duplicates=5)
                                    """

                                if (hider.n_games == MAX_GENERATION or seeker.n_games == MAX_GENERATION): break


if __name__ == "__main__":
    main()