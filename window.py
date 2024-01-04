import pygame as pg
from objects import Cell, Floor, Wall, Hider, Seeker

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
        self.cells = self.init_cells()
        self.players = self.init_players()

        # define a clock to control the fps
        self.clock = pg.time.Clock()

    def init_cells(self):
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

    def draw_map(self):
        # blit all the cells to the screen with a certain border width
        for cell in self.cells:
            pg.draw.rect(self.screen, cell.color, (cell.x, cell.y, cell.size, cell.size), 25)

    def draw_players(self):
        for player in self.players:
            pg.draw.rect(self.screen, player.color, (player.x, player.y, player.size, player.size), 0)

    def control_players(self):
        # get the keys that are currently pressed
        available_positions = self.check_available_positions()
        keys = pg.key.get_pressed()
        if keys[pg.K_UP] and available_positions[0][2]:
            self.players[0].keyboard_move('u')
        elif keys[pg.K_DOWN] and available_positions[0][3]:
            self.players[0].keyboard_move('d')
        elif keys[pg.K_LEFT] and available_positions[0][0]:
            self.players[0].keyboard_move('l')
        elif keys[pg.K_RIGHT] and available_positions[0][1]:
            self.players[0].keyboard_move('r')

        if keys[pg.K_w] and available_positions[1][2]:
            self.players[1].keyboard_move('u')
        elif keys[pg.K_s] and available_positions[1][3]:
            self.players[1].keyboard_move('d')
        elif keys[pg.K_a] and available_positions[1][0]:
            self.players[1].keyboard_move('l')
        elif keys[pg.K_d] and available_positions[1][1]:
            self.players[1].keyboard_move('r')
    
    def check_available_positions(self):
        positions = []
        for p in self.players:
            av = [False,False,False,False]
            # av : 0 - sx , 1 - dx , 2 - u , 3 - d

            i = p.x // self.size
            j = p.y // self.size

            idx = i + j * self.cols

            if p.x > 0 and self.cells[idx-1].obj_type != "wall":
                av[0] = True
            
            if p.x < self.width - self.size and self.cells[idx+1].obj_type != "wall":
                av[1] = True

            if p.y > 0 and self.cells[idx - self.cols].obj_type != "wall":
                av[2] = True
            
            if p.y < self.height - self.size and self.cells[idx + self.cols].obj_type != "wall":
                av[3] = True

            positions.append(av)
        return positions