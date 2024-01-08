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
        self.players = self.init_players()

        # define a clock to control the fps
        self.clock = pg.time.Clock()

    def init_map(self):
        cells = []
        for row in range(self.rows):
            for col in range(self.cols):
                x = col * self.size
                y = row * self.size
                if col == self.cols // 2:
                    cells.append(Wall(x, y, self.size))
                # elif col == self.cols // 3 and row == self.rows // 3:
                elif col % 3 == 0 and row % 5 == 0:
                    cells.append(MovableWall(x, y, self.size))
                else:
                    cells.append(Floor(x, y, self.size))
        return cells

    def init_players(self):
        players = []
        players.append(Hider(0, 0, self.size, map=self.map, cols=self.cols))
        players.append(Seeker(self.width - self.size, self.height - self.size, self.size, map=self.map, cols=self.cols))
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
            print(self.map[0].obj_type)

    def update(self):
        #self.check_map_integrity()
        #self.draw_map()
        self.control_players()
        self.check_map_integrity()
        self.draw_map()
        self.draw_players()

    def draw_map(self):
        # blit all the map to the screen with a certain border width
        for cell in self.map:
            pg.draw.rect(self.screen, cell.color, (cell.x, cell.y, cell.size, cell.size), 25)

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
        if keys[pg.K_LCTRL]:
            if not self.check_movable_wall(self.players[0]):
                self.players[0].movable_wall = None
                self.players[0].movable_wall_side = None

        # log player 0 movable wall position
        if self.players[0].movable_wall is not None:
            print(self.players[0].movable_wall.x // self.size, self.players[0].movable_wall.y // self.size)

        # control player 1 movable wall
        if keys[pg.K_RCTRL]:
            if not self.check_movable_wall(self.players[1]):
                self.players[1].movable_wall = None
                self.players[1].movable_wall_side = None

    def check_available_positions(self):
        positions = []
        block = "movable_wall wall"
        for p in self.players:
            av = [False, False, False, False]
            # av : 0 - sx , 1 - dx , 2 - u , 3 - d

            i = p.x // self.size
            j = p.y // self.size

            idx = i + j * self.cols
            if p.movable_wall_side == 'l' and p.x > self.size and self.map[idx - 2].obj_type not in block:
                av[0] = True
            elif p.movable_wall_side != 'l' and p.x > 0 and self.map[idx - 1].obj_type not in block:
                av[0] = True

            if av[0] == True and p.movable_wall_side == 'u' and self.map[
                idx - self.cols - 1].obj_type in block:
                av[0] = False

            if av[0] == True and p.movable_wall_side == 'd' and self.map[
                idx + self.cols - 1].obj_type in block:
                av[0] = False

            if p.movable_wall_side == 'r' and p.x < self.width - 2 * self.size and self.map[
                idx + 2].obj_type not in block:
                av[1] = True
            elif p.movable_wall_side != 'r' and p.x < self.width - self.size and self.map[
                idx + 1].obj_type not in block:
                av[1] = True

            if av[1] == True and p.movable_wall_side == 'u' and self.map[
                idx - self.cols + 1].obj_type in block:
                av[1] = False

            if av[1] == True and p.movable_wall_side == 'd' and self.map[
                idx + self.cols + 1].obj_type in block:
                av[1] = False

            if p.movable_wall_side == 'u' and p.y > self.size and self.map[idx - 2 * self.cols].obj_type not in block:
                av[2] = True
            elif p.movable_wall_side != 'u' and p.y > 0 and self.map[idx - self.cols].obj_type not in block:
                av[2] = True

            if av[2] == True and p.movable_wall_side == 'l' and self.map[
                idx - self.cols - 1].obj_type in block:
                av[2] = False

            if av[2] == True and p.movable_wall_side == 'r' and self.map[
                idx - self.cols + 1].obj_type in block:
                av[2] = False

            if p.movable_wall_side == 'd' and p.y < self.height - 2 * self.size and self.map[
                idx + 2 * self.cols].obj_type not in block:
                av[3] = True
            elif p.movable_wall_side != 'd' and p.y < self.height - self.size and self.map[
                idx + self.cols].obj_type not in block:
                av[3] = True

            if av[3] == True and p.movable_wall_side == 'l' and self.map[
                idx + self.cols - 1].obj_type in block:
                av[3] = False

            if av[3] == True and p.movable_wall_side == 'r' and self.map[
                idx - self.cols + 1].obj_type in block:
                av[3] = False

            positions.append(av)
        return positions

    def check_movable_wall(self, player):
        i = player.x // self.size
        j = player.y // self.size

        idx = i + j * self.cols

        side = None

        if player.view == "u":
            idx -= self.cols
            side = 'u'
        elif player.view == "d":
            idx += self.cols
            side = 'd'
        elif player.view == "l":
            idx -= 1
            side = 'l'
        elif player.view == "r":
            idx += 1
            side = 'r'

        if self.map[idx].obj_type == "movable_wall":
            if player.movable_wall is None:
                player.movable_wall = self.map[idx]
                player.movable_wall_side = side
            else:
                player.movable_wall = None
                player.movable_wall_side = None
            return True
        else:
            return False
        
    def check_map_integrity(self):
        for p in self.players:
            if p.movable_wall is not None:
                px, py = p.movable_wall.prev_x, p.movable_wall.prev_y
                pi = px // self.size
                pj = py // self.size
                pidx = pi + pj * self.cols
                x, y = p.movable_wall.x, p.movable_wall.y
                i = x // self.size
                j = y // self.size
                idx = i + j * self.cols
                if pidx != idx:
                    self.map[idx] = p.movable_wall
                    self.map[pidx] = Floor(px, py, self.size)