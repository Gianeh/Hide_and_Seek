import pygame as pg
from objects import Cell, Floor, Wall, MovableWall, Hider, Seeker
import numpy as np
import sys

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

class Game:
    def __init__(self, rows, cols, size):
        self.rows = rows
        self.cols = cols
        self.size = size
        self.width = cols * size
        self.height = rows * size
        pg.init()                                               #<--------
        pg.display.set_caption('HIDE AND SEEK GAME')            #<--------
        self.screen = pg.display.set_mode((self.width, self.height))
        self.screen.fill(WHITE)

        # init game variables
        self.map = self.init_map()
        self.players = self.init_players()

        # define a clock to control the fps
        self.clock = pg.time.Clock()

        self.hider = pg.image.load('img3.png')
        self.seeker = pg.image.load('img2.png')

        # crop to size
        self.hider = pg.transform.scale(self.hider, (self.size, self.size))
        self.seeker = pg.transform.scale(self.seeker, (self.size, self.size))
        # self.player_img.set_colorkey(WHITE)                     #<------------
        self.wall_img = pg.image.load('wall.png')               #<------------
        # crop to size
        self.wall_img = pg.transform.scale(self.wall_img, (self.size, self.size))

    def reset(self):
        self.map = self.init_map()
        self.players = self.init_players()


    def init_map(self):
        map = []
        for row in range(self.rows):
            map.append([])
            for col in range(self.cols):
                x = col * self.size
                y = row * self.size
                if col % 3 == 0 and row % 5 == 0:
                    map[row].append(MovableWall(x, y, self.size))
                else:
                    map[row].append(Floor(x, y, self.size))
        return map

    def init_players(self):
        players = []
        players.append(Hider(300, 300, self.size, map=self.map, cols=self.cols))
        players[0].look()
        players.append(Seeker(self.width - self.size - 300, self.height - self.size - 300, self.size, map=self.map, cols=self.cols))
        players[1].look()
        return players

    def update(self):
        self.draw_map()
        self.draw_players()
        self.draw_players_view()

    def draw_map(self):
        # blit all the map to the screen with a certain border width
        for row in range(self.rows):
            for col in range(self.cols):
                if self.map[row][col].obj_type == 'movable_wall':
                    self.screen.blit(self.wall_img, (self.map[row][col].x, self.map[row][col].y))
                else:
                    pg.draw.rect(self.screen, self.map[row][col].color, (self.map[row][col].x, self.map[row][col].y, self.map[row][col].size, self.map[row][col].size), 25)

    def draw_players_view(self):
        for p in self.players:
            #p.look()
            for l in range(len(p.view)):
                for c in range(len(p.view[l])):
                    if p.view[l][c] is None:
                        continue
                    elif p.view[l][c].obj_type == 'movable_wall':
                        self.screen.blit(self.wall_img, (p.view[l][c].x, p.view[l][c].y))
                    #elif p.view[l][c].obj_type == 'wall':
                        #pg.draw.rect(self.screen, p.view[l][c].color, (p.view[l][c].x, p.view[l][c].y, p.view[l][c].size, p.view[l][c].size), 25)
                    elif p.view[l][c].obj_type == 'hider':
                        self.screen.blit(self.hider, (p.view[l][c].x, p.view[l][c].y))
                    elif p.view[l][c].obj_type == 'seeker':
                        self.screen.blit(self.seeker, (p.view[l][c].x, p.view[l][c].y))
                    else:
                        pg.draw.rect(self.screen, (128+l*30,128+l*30,128+l*30), (p.view[l][c].x, p.view[l][c].y, p.view[l][c].size, p.view[l][c].size), 25)

    def draw_players(self):
        for p in self.players:
            if p.obj_type == 'hider':
                self.screen.blit(self.hider, (p.x, p.y))
            else:
                self.screen.blit(self.seeker, (p.x, p.y))
    
    def control_player(self, player, action):
        available_positions = self.check_available_positions(player)

        # get maximum value's index from network output
        action = np.argmax(action)
        if action == 0 and available_positions['u']:
            player.keyboard_move('u')
        elif action == 1 and available_positions['d']:
            player.keyboard_move('d')
        elif action == 2 and available_positions['sx']:
            player.keyboard_move('l')
        elif action == 3 and available_positions['dx']:
            player.keyboard_move('r')
        elif action == 4:
            if not self.check_movable_wall(player):
                player.movable_wall = None
                player.movable_wall_side = None

    def reward(self, player, gameover):
        # look a second time to update the view in order to calculate reward and seen variable coherently
        player.look()
        reward = 0

        if player.obj_type == 'seeker':
            # let seeker see
            player.see()

            # seeker wins!
            if gameover:
                reward += 100
            reward += player.seen * 4
            reward -= 1 if player.seen == 0 else 0
        
        else: # hider
            # let hider see
            player.see()

            # hider loses!
            if gameover:
                reward -= 100
            reward -= player.seen * 4
            reward += 1 if player.seen == 0 else 0

        player.reward += reward
        return reward
        
    def control_players(self):
        # get the keys that are currently pressed
        available_positions = self.check_available_positions()
        keys = pg.key.get_pressed()

        # Control player 0 movement
        if keys[pg.K_w] and available_positions[0][2]:
            self.players[0].keyboard_move('u')
        elif keys[pg.K_w]:
            self.players[0].change_direction('u')

        elif keys[pg.K_s] and available_positions[0][3]:
            self.players[0].keyboard_move('d')
        elif keys[pg.K_s]:
            self.players[0].change_direction('d')

        elif keys[pg.K_a] and available_positions[0][0]:
            self.players[0].keyboard_move('l')
        elif keys[pg.K_a]:
            self.players[0].change_direction('l')

        elif keys[pg.K_d] and available_positions[0][1]:
            self.players[0].keyboard_move('r')
        elif keys[pg.K_d]:
            self.players[0].change_direction('r')

        elif keys[pg.K_LCTRL]:
            if not self.check_movable_wall(self.players[0]):
                self.players[0].movable_wall = None
                self.players[0].movable_wall_side = None

        # control player 1 movement
        elif keys[pg.K_UP] and available_positions[1][2]:
            self.players[1].keyboard_move('u')
        elif keys[pg.K_UP]:
            self.players[1].change_direction('u')

        elif keys[pg.K_DOWN] and available_positions[1][3]:
            self.players[1].keyboard_move('d')
        elif keys[pg.K_DOWN]:
            self.players[1].change_direction('d')

        elif keys[pg.K_LEFT] and available_positions[1][0]:
            self.players[1].keyboard_move('l')
        elif keys[pg.K_LEFT]:
            self.players[1].change_direction('l')

        elif keys[pg.K_RIGHT] and available_positions[1][1]:
            self.players[1].keyboard_move('r')
        elif keys[pg.K_RIGHT]:
            self.players[1].change_direction('r')

        elif keys[pg.K_RCTRL]:
            if not self.check_movable_wall(self.players[1]):
                self.players[1].movable_wall = None
                self.players[1].movable_wall_side = None
    
    def check_available_positions(self, p):
        av = {'sx': False, 'dx': False, 'u':False, 'd':False}
        # av : 0 - sx , 1 - dx , 2 - u , 3 - d
        block = "movable_wall wall hider seeker"

        i = p.y // self.size
        j = p.x // self.size

        if p.movable_wall_side == 'l' and p.x > self.size and self.map[i][j-2].obj_type not in block:
            av['sx'] = True
        elif p.movable_wall_side != 'l' and p.x > 0 and self.map[i][j-1].obj_type not in block:
            av['sx'] = True

        if av['sx'] == True and p.movable_wall_side == 'u' and self.map[i-1][j-1].obj_type in block:
            av['sx'] = False

        if av['sx'] == True and p.movable_wall_side == 'd' and self.map[i+1][j-1].obj_type in block:
            av['sx'] = False

        if p.movable_wall_side == 'r' and p.x < self.width - 2 * self.size and self.map[i][j+2].obj_type not in block:
            av['dx'] = True
        elif p.movable_wall_side != 'r' and p.x < self.width - self.size and self.map[i][j+1].obj_type not in block:
            av['dx'] = True

        if av['dx'] == True and p.movable_wall_side == 'u' and self.map[i-1][j+1].obj_type in block:
            av['dx'] = False

        if av['dx'] == True and p.movable_wall_side == 'd' and self.map[i+1][j+1].obj_type in block:
            av['dx'] = False

        if p.movable_wall_side == 'u' and p.y > self.size and self.map[i-2][j].obj_type not in block:
            av['u'] = True
        elif p.movable_wall_side != 'u' and p.y > 0 and self.map[i-1][j].obj_type not in block:
            av['u'] = True

        if av['u'] == True and p.movable_wall_side == 'l' and self.map[i-1][j-1].obj_type in block:
            av['u'] = False

        if av['u'] == True and p.movable_wall_side == 'r' and self.map[i-1][j+1].obj_type in block:
            av['u'] = False

        if p.movable_wall_side == 'd' and p.y < self.height - 2 * self.size and self.map[i+2][j].obj_type not in block:
            av['d'] = True
        elif p.movable_wall_side != 'd' and p.y < self.height - self.size and self.map[i+1][j].obj_type not in block:
            av['d'] = True

        if av['d'] == True and p.movable_wall_side == 'l' and self.map[i+1][j-1].obj_type in block:
            av['d'] = False

        if av['d'] == True and p.movable_wall_side == 'r' and self.map[i+1][j+1].obj_type in block:
            av['d'] = False

        return av

    def check_movable_wall(self, player):
        i = player.y // self.size
        j = player.x // self.size

        side = None

        if player.direction == "u":
            if i == 0: return False
            i -= 1
            side = 'u'
        elif player.direction == "d":
            if i == self.rows - 1: return False
            i += 1
            side = 'd'
        elif player.direction == "l":
            if j == 0: return False
            j -= 1
            side = 'l'
        elif player.direction == "r":
            if j == self.cols - 1: return False
            j += 1
            side = 'r'

        if self.map[i][j].obj_type == "movable_wall":
            if player.movable_wall is None:
                player.movable_wall = self.map[i][j]
                player.movable_wall_side = side
            else:
                player.movable_wall = None
                player.movable_wall_side = None
            return True
        else:
            return False
    # a function that logs the obj_type of pointed cell
    def log_cell(self):
        pos = pg.mouse.get_pos()
        i = pos[0] // self.size
        j = pos[1] // self.size
        idx = i + j * self.cols
        print(self.map[idx].obj_type + f", coordinate matriciali: {i}, {j}")