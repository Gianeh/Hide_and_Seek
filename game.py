import pygame as pg
from objects import Cell, Floor, Wall, MovableWall, Hider, Seeker
import numpy as np
import random
import math

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Game class defines core methods, rules and rewards for the game environment
class Game:
    def __init__(self, rows, cols, size, map_name="map0.txt", random_spawn=False):
        # Environment parameters
        self.rows = rows
        self.cols = cols
        self.size = size
        self.width = cols * size
        self.height = rows * size

        # Map txt file name
        self.map_name = map_name

        # Random spawn players as opposed to the fixed starting position
        self.random_spawn = random_spawn

        # Initialize pygame and the game window
        pg.init()
        pg.display.set_caption('HIDE AND SEEK GAME')            
        self.screen = pg.display.set_mode((self.width, self.height))
        self.screen.fill(WHITE)

        # Initialize Map and Players
        self.map = self.init_map()
        self.players = self.init_players()

        # Pygame clock to control the framerate
        self.clock = pg.time.Clock()

        # Load players' sprites
        self.hider = pg.image.load('./img/hider.png')
        self.seeker = pg.image.load('./img/seeker.png')

        # Crop to correct size
        self.hider = pg.transform.scale(self.hider, (self.size, self.size))
        self.seeker = pg.transform.scale(self.seeker, (self.size, self.size))

        # Load map's sprites
        self.movable_wall_img = pg.image.load('./img/movable_wall.png')
        self.wall_img = pg.image.load('./img/wall.png')   
        self.floor_img = pg.image.load('./img/floor3.png') 

        # Crop to correct size
        self.movable_wall_img = pg.transform.scale(self.movable_wall_img, (self.size, self.size))  
        self.wall_img = pg.transform.scale(self.wall_img, (self.size, self.size))
        self.floor_img = pg.transform.scale(self.floor_img, (self.size, self.size))

    # Reset the game environment
    def reset(self):
        self.map = self.init_map()
        self.players = self.init_players()

    # Initialize the map from a txt file
    def init_map(self):
        map = []
        with open('./maps/'+self.map_name, 'r') as file:
            matrix = [list(line.strip()) for line in file]      # a matrix of file's characters
            for row in range(len(matrix)):
                map.append([])
                for col in range(len(matrix[row])):
                    x = col * self.size
                    y = row * self.size

                    # Map.txt is composed of f: floor, m: movable wall, w: wall
                    if matrix[row][col] == 'f':
                        map[row].append(Floor(x, y, self.size))
                    elif matrix[row][col] == 'm':
                        map[row].append(MovableWall(x, y, self.size))
                    elif matrix[row][col] == 'w':
                        map[row].append(Wall(x, y, self.size))
        return map
    
    # Initialize the players
    def init_players(self):
        players = []
        if self.random_spawn:
            i = random.randint(0, self.rows - 1)
            j = random.randint(0, self.cols - 1)

            while self.map[i][j].obj_type != 'floor':
                i = random.randint(0, self.rows - 1)
                j = random.randint(0, self.cols - 1)
            players.append(Hider(j*self.size, i*self.size, self.size, map=self.map))

            while self.map[i][j].obj_type != 'floor':
                i = random.randint(0, self.rows - 1)
                j = random.randint(0, self.cols - 1)
            players.append(Seeker(j*self.size, i*self.size, self.size, map=self.map))

        else: # Fixed spawn at 0,0 and rows-1,cols-1
            players.append(Hider(0, 0, self.size, map=self.map))
            players.append(Seeker((self.rows-1)*self.size, (self.cols-1)*self.size, self.size, map=self.map))


        players[0].look()
        players[1].look()
        return players

    # Render the game environment
    def update(self, lidar=True, view=True, scores=True):
        self.draw_map()
        if lidar: self.draw_lidar_view()
        if view: self.draw_players_view()
        self.draw_players()
        if scores: self.draw_scores()
    
    # Render Map
    def draw_map(self):
        for row in range(self.rows):
            for col in range(self.cols):
                if self.map[row][col].obj_type == 'movable_wall':
                    self.screen.blit(self.movable_wall_img, (self.map[row][col].x, self.map[row][col].y))
                elif self.map[row][col].obj_type == 'wall':
                    self.screen.blit(self.wall_img, (self.map[row][col].x, self.map[row][col].y))
                elif self.map[row][col].obj_type == 'floor':
                    self.screen.blit(self.floor_img, (self.map[row][col].x, self.map[row][col].y))

    # Render Players' view - with mask policy
    def draw_players_view(self):
        for p in self.players:
            for l in range(len(p.view)):
                for c in range(len(p.view[l])):
                    if p.view[l][c] is None:
                        continue
                    elif p.view[l][c].obj_type == 'movable_wall':
                        self.screen.blit(self.movable_wall_img, (p.view[l][c].x, p.view[l][c].y))
                    elif p.view[l][c].obj_type == 'wall':
                        self.screen.blit(self.wall_img, (p.view[l][c].x, p.view[l][c].y))
                    elif p.view[l][c].obj_type == 'hider':
                        self.screen.blit(self.hider, (p.view[l][c].x, p.view[l][c].y))
                    elif p.view[l][c].obj_type == 'seeker':
                        self.screen.blit(self.seeker, (p.view[l][c].x, p.view[l][c].y))
                    else:
                        # Otherwise draw view lines in a gradient of gray to make sure they go from left to right with respect to player's direction
                        pg.draw.rect(self.screen, (128+l*30,128+l*30,128+l*30), (p.view[l][c].x, p.view[l][c].y, p.view[l][c].size, p.view[l][c].size), self.size//2)

    # Render Lidar view
    def draw_lidar_view(self):
        for p in self.players:
            p.trigger_lidar()
            for i in range(len(p.lidar_view)):
                pg.draw.rect(self.screen, p.color, (p.lidar_view[i].x, p.lidar_view[i].y, p.lidar_view[i].size, p.lidar_view[i].size), self.size//2)

    # Render Players
    def draw_players(self):
        for p in self.players:
            if p.obj_type == 'hider':
                self.screen.blit(self.hider, (p.x, p.y))
            else:
                self.screen.blit(self.seeker, (p.x, p.y))

    # Blit current scores on the screen
    def draw_scores(self):
        font = pg.font.Font(None, 36)
        hider_reward = font.render(f"Hider reward: {self.players[0].reward:.3f}", True, (0,0,0))
        seeker_reward = font.render(f"Seeker reward: {self.players[1].reward:.3f}", True, (0,0,0))
        fps = int(self.clock.get_fps())
        frame_rate = font.render(f"Fps: {fps}", True, (0,0,0))
        self.screen.blit(hider_reward, (0, 0))
        self.screen.blit(seeker_reward, (0, 20))
        self.screen.blit(frame_rate, (0, 40))
    
    # Decide next action based on input array (Agent one-hot predicted Q-Values)
    def control_player(self, player, action):
        # check available positions
        available_positions = self.check_available_positions(player)        # {'sx': False, 'dx': False, 'u':False, 'd':False}
        valid = False       # flag to check collisions and valid actions

        # get maximum value's index from network output
        action = np.argmax(action)
        if action == 0 and available_positions['u']:    # move up
            player.move('u')
            valid = True
        elif action == 1 and available_positions['d']:  # move down
            player.move('d')
            valid = True
        elif action == 2 and available_positions['sx']: # move left
            player.move('l')
            valid = True
        elif action == 3 and available_positions['dx']: # move right
            player.move('r')
            valid = True
        elif action == 4:                               # grab or release movable wall
            valid = self.check_movable_wall(player)
            if not valid:
                player.movable_wall = None
                player.movable_wall_side = None
        elif action == 5:                               # stand still
            valid = True

        return valid

    # Reward policy criteria
    def reward(self, player, valid_action, wintime, frame, max_time, criterion='default'):
        # Look a second time to update the view in order to calculate reward and seen variable coherently
        player.look()
        reward = 0

        if criterion == 'default':
            if player.obj_type == 'seeker':
                # Let seeker see
                player.see()

                # Seeker wins!
                if player.seen >= wintime:
                    reward += 100
                    print("Seeker wins!")

                # Seeker gets a penalty for each step he doesn't see the hider
                reward -= 1 if player.seen == 0 else 0

                # A penalty for an invalid action
                if not valid_action:
                    reward -= 1
        
            else: # hider
                seeker = self.players[1]
                # Let seeker see
                seeker.see()

                # Hider loses!
                if seeker.seen >= wintime:
                    reward -= 100
                    print("Hider loses!")
                
                # Hider gets a reward for each step he is not seen by the seeker
                reward += 1 if seeker.seen == 0 else 0

                # A penalty for an invalid action
                if not valid_action:
                    reward -= 1

        elif criterion == 'explore':

            if player.obj_type == 'seeker':
                # Let seeker see
                player.see()

                # Seeker wins!
                if player.seen >= wintime:
                    reward += 100
                    print("Seeker wins!")
                
                # A penalty for an invalid action
                if not valid_action:
                    reward -= 1

            else: # Hider
                seeker = self.players[1]
                # Let seeker see
                seeker.see()

                # Hider loses!
                if seeker.seen >= wintime:
                    reward -= 100
                    print("Hider loses!")
                else:
                    # Hider gets a reward for each step he is not seen by the seeker otherwise a penalty
                    reward += 0.5 if seeker.seen == 0 else -20
                
                # A penalty for an invalid action
                if not valid_action:
                    reward -= 1

        elif criterion == 'explore2':

            if player.obj_type == 'seeker':
                player.see()
                # Seeker wins!
                if player.seen >= wintime:
                    reward += 10
                    print("Seeker wins!")
                
                # A reward for seeing the hider
                elif player.seen >= 1:
                    reward += 10
                
                # A penalty for an invalid action
                if not valid_action:
                    reward -= 10

            else: # Hider
                seeker = self.players[1]
                # Let seeker see
                seeker.see()

                if seeker.seen >= wintime:
                    reward -= 10
                    print("Hider loses!")
                else:
                    # Hider gets a reward for each step he is not seen by the seeker
                    reward += 10 if seeker.seen == 0 else 0
                
                # A penalty for an invalid action
                if not valid_action:
                    reward -= 10

        elif criterion == 'distance':
            other = self.players[1] if player == self.players[0] else self.players[0]

            # Matrix coordinates of both the players
            i = player.y // self.size
            j = player.x // self.size
            i_other = other.y // self.size
            j_other = other.x // self.size

            # Euclidean distance between the two players
            distance = math.sqrt((i-i_other)**2 + (j-j_other)**2)

            if player.obj_type == 'seeker':
                # Let seeker see
                player.see()

                # Seeker wins!
                if player.seen >= wintime:
                    reward += 100
                    print("Seeker wins!")
                
                # A reward proportional to the inverse of the distance between the two players
                reward += 1/distance if distance != 0 else 0

                # A penalty for an invalid action
                if not valid_action:
                    reward -= 1
            
            else: # Hider
                # Let seeker see
                other.see()

                # Hider loses!
                if other.seen >= wintime:
                    reward -= 100
                    print("Hider loses!")
                
                # A reward proportional to the distance between the two players (the further the better) if the distance is greater than a minimum radius
                reward += distance/50 if distance > 3.0 else 0 

                # A penalty for an invalid action
                if not valid_action:
                    reward -= 1

        elif criterion == 'hybrid':
            other = self.players[1] if player == self.players[0] else self.players[0]

            # Matrix coordinates of both the players
            i = player.y // self.size
            j = player.x // self.size
            i_other = other.y // self.size
            j_other = other.x // self.size
            
            # Euclidean distance between the two players
            distance = math.sqrt((i-i_other)**2 + (j-j_other)**2)

            if player.obj_type == 'seeker':
                player.see()

                # Seeker wins!
                if player.seen >= wintime:
                    reward += 10
                    print("Seeker wins!")

                # A reward for seeing the hider
                elif player.seen >= 1:
                    reward += 10
                
                # A penalty for an invalid action
                if not valid_action:
                    reward -= 10

            else: # hider
                seeker = self.players[1]
                # let seeker see
                seeker.see()

                if seeker.seen >= wintime:
                    reward -= 10
                    print("Hider loses!")
                else:
                    # Hider gets a reward for each step he is not seen by the seeker
                    reward += 10 if seeker.seen == 0 else 0

                # A reward proportional to the distance between the two players (the further the better) if the distance is greater than a minimum radius
                reward += distance/50 if distance > 3.0 else 0
                
                # A penalty for an invalid action
                if not valid_action:
                    reward -= 10

        # The most complex criterion, only works for the Hider
        elif criterion == 'smart_evasion':
            other = self.players[1] if player == self.players[0] else self.players[0]

            # Matrix coordinates of both the players
            i = player.y // self.size
            j = player.x // self.size
            i_other = other.y // self.size
            j_other = other.x // self.size

            # Euclidean distance between the two players
            distance = math.sqrt((i-i_other)**2 + (j-j_other)**2)

            # Heuristics for reward calculation
            max_distance = 26  # Side of a 26x26 map
            max_distance_reward_scale = 10  # Scale reward based on distance
            critical_distance = 5  # Minimum distance to be considered critical
            critical_distance_penalty = 50  # Critical distance penalty
            survival_reward_increment = 0.1  # Survival reward increment
            invalid_action_penalty = 1  # Non valid action penalty
            successful_evasion_bonus = 20  # Winning bonus

            # HIDER ONLY POLICY
            if player.obj_type == 'hider':
                
                distance_reward = min(distance, max_distance) / max_distance_reward_scale
                reward += distance_reward

                # Critical distance penalty
                if distance < critical_distance:
                    reward -= critical_distance_penalty

                # Survival reward increment
                if other.seen == 0:
                    reward += survival_reward_increment

                # Evasion reward
                if frame >= max_time-1 and other.seen == 0:
                    reward += successful_evasion_bonus

            elif player.obj_type == 'seeker':
                pass

            # A penalty for an invalid action
            if not valid_action:
                reward -= invalid_action_penalty


        # Update player's reward just for log purposes
        player.reward += reward
        # Return actual (single step) reward for training purposes
        return reward   
    
    # Constraints for the player's movement
    def check_available_positions(self, p):
        av = {'sx': False, 'dx': False, 'u':False, 'd':False}
        # av : 0 - sx , 1 - dx , 2 - u , 3 - d
        block = "movable_wall wall hider seeker"

        i = p.y // self.size
        j = p.x // self.size

        # (Bad) Hardcoded algebra to check available action for the player based on position and movable wall attached to player

        # Going left action is available
        if p.movable_wall_side == 'l' and p.x > self.size and self.map[i][j-2].obj_type not in block:
            av['sx'] = True
        elif p.movable_wall_side != 'l' and p.x > 0 and self.map[i][j-1].obj_type not in block:
            av['sx'] = True

        # Going left action is NOT available
        if av['sx'] == True and p.movable_wall_side == 'u' and self.map[i-1][j-1].obj_type in block:
            av['sx'] = False

        if av['sx'] == True and p.movable_wall_side == 'd' and self.map[i+1][j-1].obj_type in block:
            av['sx'] = False

        # Going right action is available
        if p.movable_wall_side == 'r' and p.x < self.width - 2 * self.size and self.map[i][j+2].obj_type not in block:
            av['dx'] = True
        elif p.movable_wall_side != 'r' and p.x < self.width - self.size and self.map[i][j+1].obj_type not in block:
            av['dx'] = True

        # Going right action is NOT available
        if av['dx'] == True and p.movable_wall_side == 'u' and self.map[i-1][j+1].obj_type in block:
            av['dx'] = False

        if av['dx'] == True and p.movable_wall_side == 'd' and self.map[i+1][j+1].obj_type in block:
            av['dx'] = False

        # Going up action is available
        if p.movable_wall_side == 'u' and p.y > self.size and self.map[i-2][j].obj_type not in block:
            av['u'] = True
        elif p.movable_wall_side != 'u' and p.y > 0 and self.map[i-1][j].obj_type not in block:
            av['u'] = True

        # Going up action is NOT available
        if av['u'] == True and p.movable_wall_side == 'l' and self.map[i-1][j-1].obj_type in block:
            av['u'] = False

        if av['u'] == True and p.movable_wall_side == 'r' and self.map[i-1][j+1].obj_type in block:
            av['u'] = False

        # Going down action is available
        if p.movable_wall_side == 'd' and p.y < self.height - 2 * self.size and self.map[i+2][j].obj_type not in block:
            av['d'] = True
        elif p.movable_wall_side != 'd' and p.y < self.height - self.size and self.map[i+1][j].obj_type not in block:
            av['d'] = True

        # Going down action is NOT available
        if av['d'] == True and p.movable_wall_side == 'l' and self.map[i+1][j-1].obj_type in block:
            av['d'] = False

        if av['d'] == True and p.movable_wall_side == 'r' and self.map[i+1][j+1].obj_type in block:
            av['d'] = False

        return av

    # Check if the player is trying to grab or release a movable wall
    def check_movable_wall(self, player):
        i = player.y // self.size
        j = player.x // self.size

        side = None

        if player.direction == "u":
            if i == 0: return False             # out of bounds
            i -= 1
            side = 'u'
        elif player.direction == "d":
            if i == self.rows - 1: return False # out of bounds
            i += 1
            side = 'd'
        elif player.direction == "l":
            if j == 0: return False             # out of bounds
            j -= 1
            side = 'l'
        elif player.direction == "r":
            if j == self.cols - 1: return False # out of bounds
            j += 1
            side = 'r'

        # If pointed cell is a movable wall:
        if self.map[i][j].obj_type == "movable_wall":

            # If the player is not holding a movable wall, grab the pointed one
            if player.movable_wall is None:
                player.movable_wall = self.map[i][j]
                player.movable_wall_side = side
            
            # If the player is already holding a movable wall, release it in pointed cell
            else:
                player.movable_wall = None
                player.movable_wall_side = None

            return True         # releasing or grabbing a movable wall is a valid action
        
        else:
            return False        # not a valid action (didn't release or grab a movable wall)
            
    # Log the cell pointed by the mouse cursor
    def log_cell(self):
        pos = pg.mouse.get_pos()
        i = pos[0] // self.size
        j = pos[1] // self.size
        print(self.map[i][j].obj_type + f", coordinate matriciali: {i}, {j}")