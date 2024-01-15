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

    hide = True
    seek = True
    #window.run()

    frames = 0
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


        # let the Agents control the players
        
        if frames % 2 or gameover and hide:
            if not gameover:
                game.players[0].look()
                hider_state = hider.get_state(game, game.players[0])
                hider_action = hider.get_action(hider_state)
                game.control_player(game.players[0], hider_action)
            hider_reward = game.reward(game.players[0], gameover)
            hider_new_state = hider.get_state(game, game.players[0])
            hider.train_short_memory(hider_state, hider_action, hider_reward, hider_new_state, gameover)
            hider.remember(hider_state, hider_action, hider_reward, hider_new_state, gameover)
        
        if not frames % 2 or gameover and seek:
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

            if (hider.n_games % 10 == 0 or seeker.n_games % 10 == 0) and (hider.n_games != 0 and seeker.n_games != 0):
                if hide : hider.train_replay("abs_reward")
                if seek : seeker.train_replay("abs_reward")
            
            if (hider.n_games % 100 == 0 or seeker.n_games % 100 == 0) and (hider.n_games != 0 and seeker.n_games != 0):
                if hide : hider.clean_memory(200)
                if seek : seeker.clean_memory(200)

        # check if gameover
        if game.players[1].seen >= WINTIME:
            gameover = True
            
        if frames >= MAX_TIME:
            stop = True

        # update screen
        if render:
            game.screen.fill(WHITE)
            game.update()
            pg.display.flip()
            game.clock.tick(30)


        # log players positions
        # print("Hider: ", game.players[0].x, game.players[0].y)
        # print("Seeker: ", game.players[1].x, game.players[1].y)

        frames += 1


if __name__ == "__main__":
    main()