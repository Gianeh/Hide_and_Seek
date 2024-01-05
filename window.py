import pygame as pg
from objects import Cell, Floor, Wall, MovableWall, Hider, Seeker

WHITE = (255, 255, 255)


class Game:
    def __init__(self, rows, cols, size):
        self.rows = rows
        self.cols = cols
        self.size = size
        self.width = cols * size
        self.height = rows * size
        self.screen = pg.display.set_mode((self.width, self.height))
        self.screen.fill(WHITE)

        # init game variables
        self.map = self.init_map()
        self.movable_walls = self.init_movable_walls()
        self.players = self.init_players()

        # define a clock to control the fps
        self.clock = pg.time.Clock()

    def init_map(self):
        cells = []
        for row in range(self.rows):
            for col in range(self.cols):
                x = col * self.size
                y = row * self.size
                if col == self.cols//2:
                    cells.append(Wall(x,y,self.size))
                else:
                    cells.append(Floor(x,y,self.size))
        return cells

    def init_movable_walls(self):
        movable_walls = []
        for row in range(self.rows):
            for col in range(self.cols):
                x = col * self.size
                y = row * self.size
                if col == self.cols // 3 and row == self.rows // 3:
                    movable_walls.append(MovableWall(x,y,self.size))
                else:
                    movable_walls.append(Floor(x,y,self.size))
        return movable_walls
    
    def init_players(self):
        players = []
        players.append(Hider(0, 0, self.size))
        players.append(Seeker(self.width - self.size, self.height - self.size, self.size))
        return players

    def run(self):
        while True:
            self.clock.tick(15)
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    return
            self.screen.fill(WHITE)
            self.update()

            # flip the display buffer
            pg.display.flip()

    def update(self):
        self.draw_map()
        self.control_players()
        self.draw_players()
        self.draw_movable_walls()

    def draw_map(self):
        # blit all the map to the screen with a certain border width
        for cell in self.map:
            pg.draw.rect(self.screen, cell.color, (cell.x, cell.y, cell.size, cell.size), 25)

    def draw_movable_walls(self):
        for cell in self.movable_walls:
            if cell.obj_type == "movable_wall":
                pg.draw.rect(self.screen, cell.color, (cell.x, cell.y, cell.size, cell.size), 0)

    def draw_players(self):
        for player in self.players:
            pg.draw.rect(self.screen, player.color, (player.x, player.y, player.size, player.size), 0)

    def control_players(self):
        # get the keys that are currently pressed
        available_positions = self.check_available_positions()
        keys = pg.key.get_pressed()

        # Control player 0 movement
        if keys[pg.K_w] and available_positions[0][2]:
            self.players[0].keyboard_move('u')
        elif keys[pg.K_s] and available_positions[0][3]:
            self.players[0].keyboard_move('d')
        elif keys[pg.K_a] and available_positions[0][0]:
            self.players[0].keyboard_move('l')
        elif keys[pg.K_d] and available_positions[0][1]:
            self.players[0].keyboard_move('r')

        # control player 1 movement
        if keys[pg.K_UP] and available_positions[1][2]:
            self.players[1].keyboard_move('u')
        elif keys[pg.K_DOWN] and available_positions[1][3]:
            self.players[1].keyboard_move('d')
        elif keys[pg.K_LEFT] and available_positions[1][0]:
            self.players[1].keyboard_move('l')
        elif keys[pg.K_RIGHT] and available_positions[1][1]:
            self.players[1].keyboard_move('r')

        # control player 0 movable wall
        if keys[pg.K_RCTRL]:
            if not self.check_movable_wall(self.players[0]):
                self.players[0].movable_wall = None

        # control player 1 movable wall
        if keys[pg.K_LCTRL]:
            if not self.check_movable_wall(self.players[1]):
                self.players[1].movable_wall = None
    
    def check_available_positions(self):
        positions = []
        for p in self.players:
            av = [False,False,False,False]
            # av : 0 - sx , 1 - dx , 2 - u , 3 - d

            i = p.x // self.size
            j = p.y // self.size

            idx = i + j * self.cols

            if p.x > 0 and self.map[idx-1].obj_type != "wall":
                av[0] = True
            
            if p.x < self.width - self.size and self.map[idx+1].obj_type != "wall":
                av[1] = True

            if p.y > 0 and self.map[idx - self.cols].obj_type != "wall":
                av[2] = True
            
            if p.y < self.height - self.size and self.map[idx + self.cols].obj_type != "wall":
                av[3] = True

            positions.append(av)
        return positions
    
    def check_movable_wall(self, player):
        i = player.x // self.size
        j = player.y // self.size

        idx = i + j * self.cols

        if player.view == "u":
            idx -= self.cols
        elif player.view == "d":
            idx += self.cols
        elif player.view == "l":
            idx -= 1
        elif player.view == "r":
            idx += 1

        if self.movable_walls[idx].obj_type == "movable_wall":
            player.movable_wall = self.movable_walls[idx] if player.movable_wall is None else None
            return True
        else:
            return False
    
'''
    0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 9 0 0 0
    0 0 0 0 0 1 0 2 0 0 0
    0 0 0 0 0 0 9 9 0 0 0
    0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0
'''