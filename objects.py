import math
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Cell is the base class for all objects in the game, contains basic information for the game behavior like block size and position
# Every inheriting object will have it's own object type, which is used to identify the object in the game
class Cell:
    def __init__(self, x, y, size, color=WHITE, obj_type=None):
        self.x = x
        self.y = y
        self.size = size
        self.color = color
        self.obj_type = obj_type

# Floor tile the players can walk on
class Floor(Cell):
    def __init__(self, x, y, size, color=WHITE, obj_type='floor'):
        super().__init__(x, y, size, color, obj_type)

# Unmovable wall, players can't step on it
class Wall(Cell):
    def __init__(self, x, y, size, color=BLACK, obj_type='wall'):
        super().__init__(x, y, size, color, obj_type)

# Micmicking the original Hide and Seek from openai, Movable wall can be moved around by players to create interesting map configurations
class MovableWall(Cell):
    def __init__(self, x, y, size, color=GREEN, obj_type='movable_wall'):
        super().__init__(x, y, size, color, obj_type)

    # If a player is carrying the wall, move it to the new position
    def move(self, direction, map):
        # Compute current matrix position
        i = self.y // self.size
        j = self.x // self.size

        # Swap the wall with the new position's Floor according to direction (u, d, l, r) and update the wall's position
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

# Parent object for Hider and Seeker, contains the basic movement and vision logic plus variables to keep track of acquired movable walls
class Player(Cell):
    def __init__(self, x, y, size, color=WHITE, obj_type='player', map=None):
        super().__init__(x, y, size, color, obj_type)

        # Direction the player is facing
        self.direction = None       # u: up, d: down, l: left, r: right

        # the wall the player is carrying
        self.movable_wall = None

        # the side of the player the wall is attached to
        self.movable_wall_side = None
        
        # the game map to place the player and relative movable walls
        self.map = map

        # Place the player in the map at it's matrix position
        self.map[self.y // self.size][self.x // self.size] = self

        # Player's view matrix
        self.view = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

        # Player's lidar vision
        self.lidar = [[] for _ in range(8)]

        # Player's lidar view for drawing purposes (lidar rays)
        self.lidar_view = []

        # Number of consecutive frames the player has seen the other player (to determine if seeker wins)
        self.seen = 0

        # Cumulative reward for the player'game (log purposes)
        self.reward = 0

    # Move the player according to the faced direction, if a movable wall is attached to the player it's moved as well
    def move(self, direction):
        # Compute current matrix position
        i = self.y // self.size
        j = self.x // self.size

        # Define a new Floor to be swapped with player's current position
        self.map[i][j] = Floor(self.x, self.y, self.size)

        # Decide new Player's position according to direction (u, d, l, r)
        if direction == 'u':
            self.y -= self.size
            self.direction = 'u'
            if self.movable_wall is not None: self.movable_wall.move('u', self.map)      # Move attached movable wall
        elif direction == 'd':
            self.y += self.size
            self.direction = 'd'
            if self.movable_wall is not None: self.movable_wall.move('d', self.map)
        elif direction == 'l':
            self.x -= self.size
            self.direction = 'l'
            if self.movable_wall is not None: self.movable_wall.move('l', self.map)
        elif direction == 'r':
            self.x += self.size
            self.direction = 'r'
            if self.movable_wall is not None: self.movable_wall.move('r', self.map)

        # Compute new matrix position
        i = self.y // self.size
        j = self.x // self.size
        self.map[i][j] = self

    # Update player's direction based o last movement
    def change_direction(self, direction):
        if direction == 'u':
            self.direction = 'u'
        elif direction == 'd':
            self.direction = 'd'
        elif direction == 'l':
            self.direction = 'l'
        elif direction == 'r':
            self.direction = 'r'

    # The view is what the game policy actually use to account for the winning condition (seen >= wintime)
    def look(self):
        j = self.x // self.size
        i = self.y // self.size
        # self.view = [[0, 0, 0, 0],  line 1
        #              [0, 0, 0, 0],  line 2
        #              [0, 0, 0, 0]]  line 3
        # A 4x3 Matrix in front of the faced direction containing the objects in the map

        # The view is updated line by line starting from the leftmost one with respect to the player's direction
        if self.direction == 'u':
            for l in range(len(self.view)):
                # The line is updated cell by cell starting from the nearest one with respect to the player's position
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

        # Lines that would be out of the map are filled with None

        # Mask the view to remove objects behind walls according to game policy
        self.mask_view()
    
    # Mask the view
    def mask_view(self):
        block = "wall movable_wall hider seeker"
        for l in range(len(self.view)):
            blocked = False
            for c in range(len(self.view[l])):
                if self.view[l][c] is None: continue        # if the cell is None it doesn't have an obj_type (out of map bounds)
                if self.view[l][c].obj_type in block and not blocked:
                    blocked = True
                    continue
                if blocked: self.view[l][c] = None

    # Fill lidar vision and lidar rays for both data acquisition and drawing purposes
    def trigger_lidar(self):
        i = self.y // self.size
        j = self.x // self.size
        self.lidar_view = []        # Clear the lidar view

        # Up direction - Same column, upper rows
        row = i-1
        col = j
        while(row >= 0):
            if self.map[row][col].obj_type == 'floor':
                self.lidar_view.append(self.map[row][col])
                row -= 1
            else:
                distance = i-row-1      # number of 'hops' to reach the object
                self.lidar[0] = [self.map[row][col].obj_type, distance]         # Lidar is a list of tuples (obj_type, distance)
                break

        if len(self.lidar[0]) == 0:
            distance = i
            self.lidar[0] = ['map_edge', distance]

        # Up-Right direction - Upper rows, right columns
        row = i-1
        col = j+1
        cols_num = len(self.map[row])
        while(row >= 0 and col < cols_num):
            if self.map[row][col].obj_type == 'floor':
                self.lidar_view.append(self.map[row][col])
                row -= 1
                col += 1
            else:
                distance = math.sqrt(((i-row-1)**2)+((col-j-1)**2))
                self.lidar[1] = [self.map[row][col].obj_type, distance]
                break

        if len(self.lidar[1]) == 0:
            distance = math.sqrt((i**2)+((cols_num-j-1)**2))
            self.lidar[1] = ['map_edge', distance]

        # Right direction - Same row, right columns
        row = i
        col = j+1
        while(col < cols_num):
            if self.map[row][col].obj_type == 'floor':
                self.lidar_view.append(self.map[row][col])
                col += 1
            else:
                distance = col-j-1
                self.lidar[2] = [self.map[row][col].obj_type, distance]
                break

        if len(self.lidar[2]) == 0:
            distance = cols_num-j-1
            self.lidar[2] = ['map_edge', distance]

        # Right-Down direction - Lower rows, right columns
        row = i+1
        col = j+1
        rows_num = len(self.map)
        while(row < rows_num and col < cols_num):
            if self.map[row][col].obj_type == 'floor':
                self.lidar_view.append(self.map[row][col])
                row += 1
                col +=1
            else:
                distance = math.sqrt(((row-i-1)**2)+(col-j-1)**2)
                self.lidar[3] = [self.map[row][col].obj_type, distance]
                break

        if len(self.lidar[3]) == 0:
            distance = math.sqrt(((rows_num-i-1)**2)+(cols_num-j-1)**2)
            self.lidar[3] = ['map_edge', distance]

        # Down direction - Same column, lower rows
        row = i+1
        col = j
        while(row < rows_num):
            if self.map[row][col].obj_type == 'floor':
                self.lidar_view.append(self.map[row][col])
                row += 1
            else:
                distance = row-i-1
                self.lidar[4] = [self.map[row][col].obj_type, distance]
                break
            
        if len(self.lidar[4]) == 0:
            distance = rows_num-i-1
            self.lidar[4] = ['map_edge', distance]

        # Down-Left direction - Lower rows, left columns
        row = i+1
        col = j-1
        while(row < rows_num and col >= 0):
            if self.map[row][col].obj_type == 'floor':
                self.lidar_view.append(self.map[row][col])
                row += 1
                col -=1
            else:
                distance = math.sqrt(((row-i-1)**2)+(j-col-1)**2)
                self.lidar[5] = [self.map[row][col].obj_type, distance]
                break

        if len(self.lidar[5]) == 0:
            distance = math.sqrt(((rows_num-i-1)**2)+(j)**2)
            self.lidar[5] = ['map_edge', distance]

        # Left direction - Same row, left columns
        row = i
        col = j-1
        while(col >= 0):
            if self.map[row][col].obj_type == 'floor':
                self.lidar_view.append(self.map[row][col])
                col -= 1
            else:
                distance = j-col-1
                self.lidar[6] = [self.map[row][col].obj_type, distance]
                break

        if len(self.lidar[6]) == 0:
            distance = j
            self.lidar[6] = ['map_edge', distance]

        # Left-Up direction - Upper rows, left columns
        row = i-1
        col = j-1
        while(row >= 0 and col >= 0):
            if self.map[row][col].obj_type == 'floor':
                self.lidar_view.append(self.map[row][col])
                row -= 1
                col -= 1
            else:
                distance = math.sqrt(((i-row-1)**2)+((j-col-1)**2))
                self.lidar[7] = [self.map[row][col].obj_type, distance]
                break

        if len(self.lidar[7]) == 0:
            distance = math.sqrt((i**2)+(j**2))
            self.lidar[7] = ['map_edge', distance]
            
# Hider player class - Objective is to avoid being seen by the seeker
class Hider(Player):
    def __init__(self, x, y, size, color=BLUE, obj_type='hider', map=None):
        super().__init__(x, y, size, color, obj_type, map)
        self.direction = 'r'
    
    def see(self):
        for l in self.view:
            for c in l:
                if c is None: continue
                if c.obj_type == 'seeker':
                    self.seen += 1
                    return
        self.seen = 0

# Seeker player class - Objective is to find the hider
class Seeker(Player):
    def __init__(self, x, y, size, color=RED, obj_type='seeker', map=None):
        super().__init__(x, y, size, color, obj_type, map)
        self.direction = 'l'
    
    def see(self):
        for l in self.view:
            for c in l:
                if c is None: continue
                if c.obj_type == 'hider':
                    self.seen += 1
                    return
        self.seen = 0

