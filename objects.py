
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (255, 0 ,0)

class Cell:
    def __init__(self, x, y, size, color = WHITE, obj_type = None):
        self.x = x
        self.y = y
        self.size = size
        self.color = color
        self.obj_type = obj_type
    
class Floor(Cell):
    def __init__(self, x, y, size, color = WHITE, obj_type = 'floor'):
        super().__init__(x, y, size, color, obj_type)

class Wall(Cell):
    def __init__(self, x, y, size, color = BLACK, obj_type = 'wall'):
        super().__init__(x, y, size, color, obj_type)

class Player(Cell):
    def __init__(self, x, y, size, color = WHITE, obj_type = 'player'):
        super().__init__(x, y, size, color, obj_type)

    def keyboard_move(self, direction):
        if direction == 'u':
            self.y -= self.size
        elif direction == 'd':
            self.y += self.size
        elif direction == 'l':
            self.x -= self.size
        elif direction == 'r':
            self.x += self.size

class Hider(Player):
    def __init__(self, x, y, size, color = BLUE, obj_type = 'hider'):
        super().__init__(x, y, size, color, obj_type)


class Seeker(Player):
    def __init__(self, x, y, size, color = RED, obj_type = 'seeker'):
        super().__init__(x, y, size, color, obj_type)
