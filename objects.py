import math
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
        self.lidar = [[] for _ in range(8)]
        self.lidar_view = []
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


    def trigger_lidar(self):
        i = self.y // self.size
        j = self.x // self.size
        self.lidar_view = []
        #up direction
        row = i-1
        col = j
        while(row >= 0):
            if self.map[row][col].obj_type == 'floor':
                self.lidar_view.append(self.map[row][col])
                row -= 1
                continue
            else:
                distance = i-row-1
                self.lidar[0] = [self.map[row][col].obj_type, distance]
                break
        if len(self.lidar[0]) == 0:
            distance = i
            self.lidar[0] = ['map_edge', distance]

        #up-right direction
        row = i-1
        col = j+1
        cols_num = len(self.map[row])
        while(row >= 0 and col < cols_num):
            if self.map[row][col].obj_type == 'floor':
                self.lidar_view.append(self.map[row][col])
                row -= 1
                col += 1
                continue
            else:
                distance = math.sqrt(((i-row-1)**2)+((col-j-1)**2))
                self.lidar[1] = [self.map[row][col].obj_type, distance]
                break
        if len(self.lidar[1]) == 0:
            distance = math.sqrt((i**2)+((cols_num-j-1)**2))
            self.lidar[1] = ['map_edge', distance]

        #right direction
        row = i
        col = j+1
        while(col < cols_num):
            if self.map[row][col].obj_type == 'floor':
                self.lidar_view.append(self.map[row][col])
                col += 1
                continue
            else:
                distance = col-j-1
                self.lidar[2] = [self.map[row][col].obj_type, distance]
                break
        if len(self.lidar[2]) == 0:
            distance = cols_num-j-1
            self.lidar[2] = ['map_edge', distance]

        #right-down direction
        row = i+1
        col = j+1
        rows_num = len(self.map)
        while(row < rows_num and col < cols_num):
            if self.map[row][col].obj_type == 'floor':
                self.lidar_view.append(self.map[row][col])
                row += 1
                col +=1
                continue
            else:
                distance = math.sqrt(((row-i-1)**2)+(col-j-1)**2)
                self.lidar[3] = [self.map[row][col].obj_type, distance]
                break
        if len(self.lidar[3]) == 0:
            distance = math.sqrt(((rows_num-i-1)**2)+(cols_num-j-1)**2)
            self.lidar[3] = ['map_edge', distance]

        #down direction
        row = i+1
        col = j
        while(row < rows_num):
            if self.map[row][col].obj_type == 'floor':
                self.lidar_view.append(self.map[row][col])
                row += 1
                continue
            else:
                distance = row-i-1
                self.lidar[4] = [self.map[row][col].obj_type, distance]
                break
        if len(self.lidar[4]) == 0:
            distance = rows_num-i-1
            self.lidar[4] = ['map_edge', distance]

        #down-left direction
        row = i+1
        col = j-1
        while(row < rows_num and col >= 0):
            if self.map[row][col].obj_type == 'floor':
                self.lidar_view.append(self.map[row][col])
                row += 1
                col -=1
                continue
            else:
                distance = math.sqrt(((row-i-1)**2)+(j-col-1)**2)
                self.lidar[5] = [self.map[row][col].obj_type, distance]
                break
        if len(self.lidar[5]) == 0:
            distance = math.sqrt(((rows_num-i-1)**2)+(j)**2)
            self.lidar[5] = ['map_edge', distance]

        #left direction
        row = i
        col = j-1
        while(col >= 0):
            if self.map[row][col].obj_type == 'floor':
                self.lidar_view.append(self.map[row][col])
                col -= 1
                continue
            else:
                distance = j-col-1
                self.lidar[6] = [self.map[row][col].obj_type, distance]
                break
        if len(self.lidar[6]) == 0:
            distance = j
            self.lidar[6] = ['map_edge', distance]

        #left-up direction
        row = i-1
        col = j-1
        while(row >= 0 and col >= 0):
            if self.map[row][col].obj_type == 'floor':
                self.lidar_view.append(self.map[row][col])
                row -= 1
                col -= 1
                continue
            else:
                distance = math.sqrt(((i-row-1)**2)+((j-col-1)**2))
                self.lidar[7] = [self.map[row][col].obj_type, distance]
                break
        if len(self.lidar[7]) == 0:
            distance = math.sqrt((i**2)+(j**2))
            self.lidar[7] = ['map_edge', distance]
            



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
        self.seen = 0


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
        self.seen = 0

