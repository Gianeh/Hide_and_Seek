WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

WINTIME = 10


class Cell:
    def __init__(self, x, y, size, color=WHITE, obj_type=None):
        self.x = x
        self.y = y
        self.size = size
        self.color = color
        self.obj_type = obj_type


class Floor(Cell):
    def __init__(self, x, y, size, color=WHITE, obj_type='floor'):
        super().__init__(x, y, size, color, obj_type)


class Wall(Cell):
    def __init__(self, x, y, size, color=BLACK, obj_type='wall'):
        super().__init__(x, y, size, color, obj_type)


class MovableWall(Cell):
    def __init__(self, x, y, size, color=GREEN, obj_type='movable_wall'):
        super().__init__(x, y, size, color, obj_type)

    def move(self, direction, map, cols):
        i = self.y // self.size
        j = self.x // self.size

        if direction == 'u':
            temp = map[i-1][j]
            map[i-1][j] = map[i][j]
            map[i][j] = temp
            map[i][j].y = self.y
            self.y -= self.size
        elif direction == 'd':
            temp = map[i+1][j]
            map[i+1][j] = map[i][j]
            map[i][j] = temp
            map[i][j].y = self.y
            self.y += self.size
        elif direction == 'l':
            temp = map[i][j-1]
            map[i][j-1] = map[i][j]
            map[i][j] = temp
            map[i][j].x = self.x
            self.x -= self.size
        elif direction == 'r':
            temp = map[i][j+1]
            map[i][j+1] = map[i][j]
            map[i][j] = temp
            map[i][j].x = self.x
            self.x += self.size



class Player(Cell):
    def __init__(self, x, y, size, color=WHITE, obj_type='player', map=None, cols=None):
        super().__init__(x, y, size, color, obj_type)
        self.direction = None
        self.movable_wall = None
        self.movable_wall_side = None
        self.map = map
        # position yourself on the map
        i = self.y // self.size
        j = self.x // self.size
        self.map[i][j] = self
        self.cols = cols
        self.view = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        self.seen = 0

        self.reward = 0

    def keyboard_move(self, direction):
        i = self.y // self.size
        j = self.x // self.size
        self.map[i][j] = Floor(self.x, self.y, self.size)

        if direction == 'u':
            self.y -= self.size
            self.direction = 'u'
            if self.movable_wall is not None: self.movable_wall.move('u', self.map, self.cols)
        elif direction == 'd':
            self.y += self.size
            self.direction = 'd'
            if self.movable_wall is not None: self.movable_wall.move('d', self.map, self.cols)
        elif direction == 'l':
            self.x -= self.size
            self.direction = 'l'
            if self.movable_wall is not None: self.movable_wall.move('l', self.map, self.cols)
        elif direction == 'r':
            self.x += self.size
            self.direction = 'r'
            if self.movable_wall is not None: self.movable_wall.move('r', self.map, self.cols)

        i = self.y // self.size
        j = self.x // self.size
        self.map[i][j] = self

    def change_direction(self, direction):
        if direction == 'u':
            self.direction = 'u'
        elif direction == 'd':
            self.direction = 'd'
        elif direction == 'l':
            self.direction = 'l'
        elif direction == 'r':
            self.direction = 'r'

    def look(self):
        j = self.x // self.size
        i = self.y // self.size
        # self.view = [[0, 0, 0, 0],
        #              [0, 0, 0, 0],
        #              [0, 0, 0, 0]]

        if self.direction == 'u':
            for l in range(len(self.view)):
                for c in range(len(self.view[l])):
                    row = i-(c+1)
                    col = j+(l-1)
                    self.view[l][c] = self.map[row][col] if row >= 0 and col >= 0 and col < len(self.map[0]) else None
        elif self.direction == 'd':
            for l in range(len(self.view)):
                for c in range(len(self.view[l])):
                    row = i+(c+1)
                    col = j+1-l
                    self.view[l][c] = self.map[row][col] if row < len(self.map) and col >= 0 and col < len(self.map[0]) else None
        elif self.direction == 'l':
            for l in range(len(self.view)):
                for c in range(len(self.view[l])):
                    row = i-l+1
                    col = j-(c+1)
                    self.view[l][c] = self.map[row][col] if row >= 0 and row < len(self.map) and col >= 0 else None
        elif self.direction == 'r':
            for l in range(len(self.view)):
                for c in range(len(self.view[l])):
                    row = i+l-1
                    col = j+(c+1)
                    self.view[l][c] = self.map[row][col] if row >= 0 and row < len(self.map) and col < len(self.map[0]) else None

        self.mask_view()
    
    def mask_view(self):
        block = "wall movable_wall hider seeker"
        for l in range(len(self.view)):
            blocked = False
            for c in range(len(self.view[l])):
                if self.view[l][c] is None: continue
                if self.view[l][c].obj_type in block and not blocked:
                    blocked = True
                    continue
                if blocked: self.view[l][c] = None



class Hider(Player):
    def __init__(self, x, y, size, color=BLUE, obj_type='hider', map=None, cols=None):
        super().__init__(x, y, size, color, obj_type, map, cols)
        self.direction = 'r'
    
    def see(self):
        for l in self.view:
            for c in l:
                if c is None: continue
                if c.obj_type == 'seeker':
                    self.seen += 1
                    return
        self.seen -= 1 if self.seen > 0 else 0


class Seeker(Player):
    def __init__(self, x, y, size, color=RED, obj_type='seeker', map=None, cols=None):
        super().__init__(x, y, size, color, obj_type, map, cols)
        self.direction = 'l'
    
    def see(self):
        for l in self.view:
            for c in l:
                if c is None: continue
                if c.obj_type == 'hider':
                    self.seen += 1
                    return
        self.seen -= 1 if self.seen > 0 else 0

