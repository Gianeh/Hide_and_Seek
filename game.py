import pygame as pg
from objects import Cell, Floor, Wall, MovableWall, Hider, Seeker
import numpy as np
import random
import math

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

class Game:
    def __init__(self, rows, cols, size, map_name):
        self.rows = rows
        self.cols = cols
        self.size = size
        self.width = cols * size
        self.height = rows * size
        self.map_name = map_name
        pg.init()                                               #<--------
        pg.display.set_caption('HIDE AND SEEK GAME')            #<--------
        self.screen = pg.display.set_mode((self.width, self.height))
        self.screen.fill(WHITE)

        # init game variables
        self.map = self.init_map()
        self.players = self.init_players()

        # define a clock to control the fps
        self.clock = pg.time.Clock()

        self.hider = pg.image.load('./img/hider.png')
        self.seeker = pg.image.load('./img/seeker.png')

        # crop to size
        self.hider = pg.transform.scale(self.hider, (self.size, self.size))
        self.seeker = pg.transform.scale(self.seeker, (self.size, self.size))
        # self.player_img.set_colorkey(WHITE)                     
        self.movable_wall_img = pg.image.load('./img/movable_wall.png')               #<------------
        # crop to size
        self.movable_wall_img = pg.transform.scale(self.movable_wall_img, (self.size, self.size))
        self.wall_img = pg.image.load('./img/wall.png')   
        self.wall_img = pg.transform.scale(self.wall_img, (self.size, self.size))
        self.floor_img = pg.image.load('./img/floor3.png')   
        self.floor_img = pg.transform.scale(self.floor_img, (self.size, self.size))

    def reset(self):
        self.map = self.init_map()
        self.players = self.init_players()


    '''def init_map(self):
        map = []
        for row in range(self.rows):
            map.append([])
            for col in range(self.cols):
                x = col * self.size
                y = row * self.size
                
                if col % 6 == 0 and row % 6 == 0:
                    map[row].append(MovableWall(x, y, self.size))
                else:
                    map[row].append(Floor(x, y, self.size))
                
                
               # map[row].append(Floor(x, y, self.size))
        return map'''
    
    def init_map(self):
        map = []
        with open('./maps/'+self.map_name, 'r') as file:
            matrix = [list(line.strip()) for line in file]
            for row in range(len(matrix)):
                map.append([])
                for col in range(len(matrix[row])):
                    x = col * self.size
                    y = row * self.size
                    if matrix[row][col] == 'f':
                        map[row].append(Floor(x, y, self.size))
                    elif matrix[row][col] == 'm':
                        map[row].append(MovableWall(x, y, self.size))
                    elif matrix[row][col] == 'w':
                        map[row].append(Wall(x, y, self.size))
        return map



    def init_players(self):
        players = []
        i = random.randint(0, self.rows - 1)
        j = random.randint(0, self.cols - 1)

        while self.map[i][j].obj_type != 'floor':
            i = random.randint(0, self.rows - 1)
            j = random.randint(0, self.cols - 1)
        players.append(Hider(j*self.size, i*self.size, self.size, map=self.map, cols=self.cols))
        #print("coordinates: ", i, j ,"obj_type: ",self.map[i][j].obj_type, "\n")

        while self.map[i][j].obj_type != 'floor':
            i = random.randint(0, self.rows - 1)
            j = random.randint(0, self.cols - 1)
        players.append(Seeker(j*self.size, i*self.size, self.size, map=self.map, cols=self.cols))
        #print("coordinates: ", i, j, "obj_type: ", self.map[i][j].obj_type, "\n")

        players[0].look()
        players[1].look()
        return players

    def update(self):
        self.draw_map()
        self.draw_lidar_view()
        self.draw_players_view()
        self.draw_players()
        self.draw_scores()
        

    def draw_map(self):
        # blit all the map to the screen with a certain border width
        for row in range(self.rows):
            for col in range(self.cols):
                if self.map[row][col].obj_type == 'movable_wall':
                    self.screen.blit(self.movable_wall_img, (self.map[row][col].x, self.map[row][col].y))
                elif self.map[row][col].obj_type == 'wall':
                    self.screen.blit(self.wall_img, (self.map[row][col].x, self.map[row][col].y))
                elif self.map[row][col].obj_type == 'floor':
                    self.screen.blit(self.floor_img, (self.map[row][col].x, self.map[row][col].y))
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
                        self.screen.blit(self.movable_wall_img, (p.view[l][c].x, p.view[l][c].y))
                    elif p.view[l][c].obj_type == 'wall':
                        self.screen.blit(self.wall_img, (p.view[l][c].x, p.view[l][c].y))
                    elif p.view[l][c].obj_type == 'hider':
                        self.screen.blit(self.hider, (p.view[l][c].x, p.view[l][c].y))
                    elif p.view[l][c].obj_type == 'seeker':
                        self.screen.blit(self.seeker, (p.view[l][c].x, p.view[l][c].y))
                    else:
                        pg.draw.rect(self.screen, (128+l*30,128+l*30,128+l*30), (p.view[l][c].x, p.view[l][c].y, p.view[l][c].size, p.view[l][c].size), 25)

    def draw_lidar_view(self):
        for p in self.players:
            p.trigger_lidar()
            for i in range(len(p.lidar_view)):
                pg.draw.rect(self.screen, p.color, (p.lidar_view[i].x, p.lidar_view[i].y, p.lidar_view[i].size, p.lidar_view[i].size), 25)

    def draw_players(self):
        for p in self.players:
            if p.obj_type == 'hider':
                self.screen.blit(self.hider, (p.x, p.y))
            else:
                self.screen.blit(self.seeker, (p.x, p.y))

    def draw_scores(self):
        font = pg.font.Font(None, 36)
        hider_reward = font.render(f"Hider reward: {self.players[0].reward:.3f}", True, (0,0,0))
        seeker_reward = font.render(f"Seeker reward: {self.players[1].reward:.3f}", True, (0,0,0))
        fps = int(self.clock.get_fps())
        frame_rate = font.render(f"Fps: {fps}", True, (0,0,0))
        self.screen.blit(hider_reward, (0, 0))
        self.screen.blit(seeker_reward, (0, 20))
        self.screen.blit(frame_rate, (0, 40))
    
    def control_player(self, player, action):
        available_positions = self.check_available_positions(player)
        valid = False

        # get maximum value's index from network output
        action = np.argmax(action)
        if action == 0 and available_positions['u']:
            player.keyboard_move('u')
            valid = True
        elif action == 1 and available_positions['d']:
            player.keyboard_move('d')
            valid = True
        elif action == 2 and available_positions['sx']:
            player.keyboard_move('l')
            valid = True
        elif action == 3 and available_positions['dx']:
            player.keyboard_move('r')
            valid = True
        elif action == 4:
            valid = self.check_movable_wall(player)
            if not valid:
                player.movable_wall = None
                player.movable_wall_side = None
        elif action == 5:
            valid = True
            # stand still

        return valid

    def reward(self, player, valid_action, wintime, criterion='default'):
        # look a second time to update the view in order to calculate reward and seen variable coherently
        player.look()
        reward = 0
        if criterion == 'default':
            if player.obj_type == 'seeker':
                # let seeker see
                player.see()

                # seeker wins!
                if player.seen >= wintime:
                    reward += 100
                    print("Seeker wins!")
                    print("Seeker seen: ", player.seen)

                #reward += player.seen
                reward -= 1 if player.seen == 0 else 0

                if not valid_action:
                    reward -= 1
        
            else: # hider
                # let hider see
                player.see()    # theorically is no longer needed

                other = self.players[1] if player == self.players[0] else self.players[0]
                # let seeker see
                other.see()

                # hider loses!
                if other.seen >= wintime:
                    reward -= 100
                    print("Hider loses!")
                
                #reward -= other.seen
                reward += 1 if other.seen == 0 else 0


                if not valid_action:
                    reward -= 1

        elif criterion == 'explore':

            if player.obj_type == 'seeker':
                # let seeker see
                player.see()

                # seeker wins!
                if player.seen >= wintime:
                    reward += 100
                    print("Seeker wins!")
                    print("Seeker seen: ", player.seen)
                
                if not valid_action:
                    reward -= 1

            else: # hider
                # let hider see
                player.see()

                other = self.players[1] if player == self.players[0] else self.players[0]
                # let seeker see
                other.see()

                if other.seen >= wintime:
                    reward -= 100
                    print("Hider loses!")
                else:
                    #reward += 0.2
                    reward += 0.5 if other.seen == 0 else 0     #try to avoid being in seeker view
                
                if not valid_action:
                    reward -= 1

        elif criterion == 'explore2':

            if player.obj_type == 'seeker':
                player.see()
                # seeker wins!
                if player.seen >= wintime:
                    reward += 10
                    print("Seeker wins!")
                    print("Seeker seen: ", player.seen)
                elif player.seen >= 1:
                    reward += 10
                
                if not valid_action:
                    reward -= 10

            else: # hider
                # let hider see
                player.see()

                other = self.players[1] if player == self.players[0] else self.players[0]
                # let seeker see
                other.see()

                if other.seen >= wintime:
                    reward -= 10
                    print("Hider loses!")
                else:
                    #reward += 0.2
                    reward += 10 if other.seen == 0 else 0     #try to avoid being in seeker view
                
                if not valid_action:
                    reward -= 10


        elif criterion == 'distance':
            other = self.players[1] if player == self.players[0] else self.players[0]
            # i,j of both
            i = player.y // self.size
            j = player.x // self.size
            i_other = other.y // self.size
            j_other = other.x // self.size
            distance = math.sqrt((i-i_other)**2 + (j-j_other)**2)

            if player.obj_type == 'seeker':
                # let seeker see
                player.see()

                # seeker wins!
                if player.seen >= wintime:
                    reward += 100
                    print("Seeker wins!")
                    print("Seeker seen: ", player.seen)
                
                reward += 1/distance if distance != 0 else 0

                if not valid_action:
                    reward -= 1
            
            else: # hider

                # let hider see
                player.see()
                # let seeker see
                other.see()

                if other.seen >= wintime:
                    reward -= 100
                    print("Hider loses!")
                
                #reward -= 1/distance if distance != 0 else 0
                reward += distance/50 if distance > 3.0 else 0      #min distance radius = 3 to get a reward score (try to maintain min distance from seeker)

                if not valid_action:
                    reward -= 1

        # update player's reward just for log purposes
        player.reward += reward
        # return actual reward for training purposes
        return reward
        
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
        #idx = i + j * self.cols
        print(self.map[i][j].obj_type + f", coordinate matriciali: {i}, {j}")