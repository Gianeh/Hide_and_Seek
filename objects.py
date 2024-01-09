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
        self.cols = cols
        #self.view = [[None * 4], [None * 4], [None * 4]]

    def keyboard_move(self, direction):
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

    def change_direction(self, direction):
        if direction == 'u':
            self.direction = 'u'
        elif direction == 'd':
            self.direction = 'd'
        elif direction == 'l':
            self.direction = 'l'
        elif direction == 'r':
            self.direction = 'r'


class Hider(Player):
    def __init__(self, x, y, size, color=BLUE, obj_type='hider', map=None, cols=None):
        super().__init__(x, y, size, color, obj_type, map, cols)


class Seeker(Player):
    def __init__(self, x, y, size, color=RED, obj_type='seeker', map=None, cols=None):
        super().__init__(x, y, size, color, obj_type, map, cols)



'''
0 0 0 0 0 0 9 9 0 
0 0 0 0 0 9 9 9 0
0 0 0 1 9 5 9 9 0
0 0 0 0 0 5 9 9 0 
0 0 0 0 0 0 9 9 0 
'''