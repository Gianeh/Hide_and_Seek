WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)


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
        i = self.x // self.size
        j = self.y // self.size
        idx = i + j * cols

        if direction == 'u':
            temp = map[idx - cols]
            map[idx - cols] = map[idx]
            map[idx] = temp
            map[idx].y = self.y
            self.y -= self.size
        elif direction == 'd':
            temp = map[idx + cols]
            map[idx + cols] = map[idx]
            map[idx] = temp
            map[idx].y = self.y
            self.y += self.size
        elif direction == 'l':
            temp = map[idx - 1]
            map[idx - 1] = map[idx]
            map[idx] = temp
            map[idx].x = self.x
            self.x -= self.size
        elif direction == 'r':
            temp = map[idx + 1]
            map[idx + 1] = map[idx]
            map[idx] = temp
            map[idx].x = self.x
            self.x += self.size



class Player(Cell):
    def __init__(self, x, y, size, color=WHITE, obj_type='player', map=None, cols=None):
        super().__init__(x, y, size, color, obj_type)
        self.direction = None
        self.movable_wall = None
        self.movable_wall_side = None
        self.map = map
        # position yourself on the map
        i = self.x // self.size
        j = self.y // self.size
        idx = i + j * cols
        self.map[idx] = self
        self.cols = cols
        self.matrix = self.map_to_matrix()
        self.view = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    def keyboard_move(self, direction):
        i = self.x // self.size
        j = self.y // self.size
        idx = i + j * self.cols
        self.map[idx] = Floor(self.x, self.y, self.size)

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

        i = self.x // self.size
        j = self.y // self.size
        idx = i + j * self.cols
        self.map[idx] = self

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
        self.matrix = self.map_to_matrix()
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
                    self.view[l][c] = self.matrix[i-(c+1)][j+(l-1)] if row >= 0 and col >= 0 and col < len(self.matrix[0]) else None
        elif self.direction == 'd':
            for l in range(len(self.view)):
                for c in range(len(self.view[l])):
                    row = i+(c+1)
                    col = j+1-l
                    self.view[l][c] = self.matrix[i+(c+1)][j+1-l] if row < len(self.matrix) and col >= 0 and col < len(self.matrix[0]) else None
        elif self.direction == 'l':
            for l in range(len(self.view)):
                for c in range(len(self.view[l])):
                    row = i+l-1
                    col = j-(c+1)
                    self.view[l][c] = self.matrix[i+l-1][j-(c+1)] if row >= 0 and row < len(self.matrix) and col >= 0 else None
        elif self.direction == 'r':
            for l in range(len(self.view)):
                for c in range(len(self.view[l])):
                    row = i+1-l
                    col = j+(c+1)
                    self.view[l][c] = self.matrix[i+1-l][j+(c+1)] if row >= 0 and row < len(self.matrix) and col < len(self.matrix[0]) else None


    def map_to_matrix(self):
        matrix = []
        rows = len(self.map) // self.cols
        for i in range(rows):
            matrix.append([])
            for j in range(self.cols):
                matrix[i].append(self.map[i*self.cols + j])
        
        return matrix



class Hider(Player):
    def __init__(self, x, y, size, color=BLUE, obj_type='hider', map=None, cols=None):
        super().__init__(x, y, size, color, obj_type, map, cols)
        self.direction = 'r'


class Seeker(Player):
    def __init__(self, x, y, size, color=RED, obj_type='seeker', map=None, cols=None):
        super().__init__(x, y, size, color, obj_type, map, cols)
        self.direction = 'l'



'''
0 0 0 0 0 0 9 9 0 
0 0 0 0 0 9 9 9 0
0 0 0 1 9 5 9 1 0
0 0 0 0 0 5 9 9 0 
0 0 0 0 0 0 9 9 0 
'''