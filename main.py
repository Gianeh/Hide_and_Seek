# Hide and Seek in a grid!
import pygame as pg
from window import Game
from agent import Agent
import sys

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

MAX_TIME = 1000
WINTIME = 20


def main():
    # start game!
    game = Game(20,20,30)
    hider = Agent('hider')
    seeker = Agent('seeker')
    #window.run()

    frames = 0
    gameover = False
    stop = False
    while True:
        # close the window
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()

        # let the Agents control the players
        
        if frames % 2 or gameover:
            if not gameover:
                game.players[0].look()
                hider_state = hider.get_state(game, game.players[0])
                hider_action = hider.get_action(hider_state)
                game.control_player(game.players[0], hider_action)
            hider_reward = game.reward(game.players[0], gameover)
            hider_new_state = hider.get_state(game, game.players[0])
            hider.train_short_memory(hider_state, hider_action, hider_reward, hider_new_state, gameover)
            hider.remember(hider_state, hider_action, hider_reward, hider_new_state, gameover)
        
        if not frames % 2 or gameover:
            if not gameover:
                game.players[1].look()
                seeker_state = seeker.get_state(game, game.players[1])
                seeker_action = seeker.get_action(seeker_state)
                game.control_player(game.players[1], seeker_action)
            seeker_reward = game.reward(game.players[1], gameover)
            seeker_new_state = seeker.get_state(game, game.players[1])
            seeker.train_short_memory(seeker_state, seeker_action, seeker_reward, seeker_new_state, gameover)
            seeker.remember(seeker_state, seeker_action, seeker_reward, seeker_new_state, gameover)

        if gameover or stop:

            # log some info avout generation
            print('Generation: ', seeker.n_games, ' Epsilon: ', seeker.epsilon,
                '\nSeeker Reward: ', game.players[1].reward,
                '\nHider Reward: ', game.players[0].reward)

            game.reset()
            frames = 0
            gameover = False
            stop = False
            hider.n_games += 1
            hider.train_long_memory()
            seeker.n_games += 1
            seeker.train_long_memory()

        # check if gameover
        if game.players[1].seen >= WINTIME:
            gameover = True
            
        if frames >= MAX_TIME:
            stop = True

        # actual game loop
        game.screen.fill(WHITE)
        game.update()
        pg.display.flip()
        # game.log_cell()
        game.clock.tick(144)


        # log players positions
        # print("Hider: ", game.players[0].x, game.players[0].y)
        # print("Seeker: ", game.players[1].x, game.players[1].y)

        frames += 1


if __name__ == "__main__":
    main()