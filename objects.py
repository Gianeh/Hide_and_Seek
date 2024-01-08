WHITE = (120, 120, 120)
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
        self.prev_x = x
        self.prev_y = y

    def move(self, direction):

        if direction == 'u':
            self.prev_y = self.y
            self.y -= self.size
        elif direction == 'd':
            self.prev_y = self.y
            self.y += self.size
        elif direction == 'l':
            self.prev_x = self.x
            self.x -= self.size
        elif direction == 'r':
            self.prev_x = self.x
            self.x += self.size


class Player(Cell):
    def __init__(self, x, y, size, color=WHITE, obj_type='player'):
        super().__init__(x, y, size, color, obj_type)
        self.direction = None
        self.movable_wall = None
        self.movable_wall_side = None
        self.view = [[None * 4], [None * 4], [None * 4]]

    def keyboard_move(self, direction):
        if direction == 'u':
            self.y -= self.size
            self.direction = 'u'
            if self.movable_wall is not None: self.movable_wall.move('u')
        elif direction == 'd':
            self.y += self.size
            self.direction = 'd'
            if self.movable_wall is not None: self.movable_wall.move('d')
        elif direction == 'l':
            self.x -= self.size
            self.direction = 'l'
            if self.movable_wall is not None: self.movable_wall.move('l')
        elif direction == 'r':
            self.x += self.size
            self.direction = 'r'
            if self.movable_wall is not None: self.movable_wall.move('r')

        # print(f"I'm looking {self.view}")


class Hider(Player):
    def __init__(self, x, y, size, color=BLUE, obj_type='hider'):
        super().__init__(x, y, size, color, obj_type)


class Seeker(Player):
    def __init__(self, x, y, size, color=RED, obj_type='seeker'):
        super().__init__(x, y, size, color, obj_type)



'''
0 0 0 0 0 0 9 9 0 
0 0 0 0 0 9 9 9 0
0 0 0 1 9 5 9 9 0
0 0 0 0 0 5 9 9 0 
0 0 0 0 0 0 9 9 0 
'''