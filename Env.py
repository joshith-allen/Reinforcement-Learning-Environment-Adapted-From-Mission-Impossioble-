import pygame
import numpy as np
import sys
import gymnasium as gym

class PadmEnvPygame(gym.Env):
    def __init__(self, grid_size=7, goal_coordinates=(3,6))-> None:
        super(PadmEnvPygame, self).__init__()
        self.grid_size = grid_size
        self.cell_size = 64
        self.agent_state = None
        self.reward = 0
        self.info = {}
        self.goal = np.array(goal_coordinates)
        self.done = False
        self.obstacle_states = []
        self.random_initialization = False  # If True, the agent will be initialized randomly

        # Action-space:
        self.action_space = gym.spaces.Discrete(4)

         # Observation space:
        self.observation_space = gym.spaces.Box(low=0, high=self.grid_size, shape=(2,), dtype=np.int32)

        # Initialize the window:
        pygame.init()
        self.screen = pygame.display.set_mode((self.cell_size*self.grid_size, self.cell_size*self.grid_size))
        self.screen_size = self.grid_size * self.cell_size

        pygame.display.set_caption("PadmEnv with Pygame")
        self.clock = pygame.time.Clock()

        self.points = [[5, 1]]  # Points to collect

        self.agent_img = pygame.image.load("/home/josith/Documents/PADM/agent_hunt.jpg")
        self.goal_img = pygame.image.load("/home/josith/Documents/PADM/goal_entity.jpg")
        self.obstacle_img = pygame.image.load("/home/josith/Documents/PADM/obstacle_explosion.jpg")
        self.points_img = pygame.image.load("/home/josith/Documents/PADM/reward_keys.jpg")

        self.agent_img = pygame.transform.scale(self.agent_img, (self.cell_size, self.cell_size))
        self.goal_img = pygame.transform.scale(self.goal_img, (self.cell_size, self.cell_size))
        self.obstacle_img = pygame.transform.scale(self.obstacle_img, (self.cell_size, self.cell_size))
        self.points_img = pygame.transform.scale(self.points_img, (self.cell_size, self.cell_size))


    def reset(self):
        self.agent_state = np.array([1, 1])
        self.score = 0
        self.done = False
        self.reward = 0
        self.points = [[5, 1]]

        if self.random_initialization:
           self.agent_state = np.array([np.random.choice([0,1,2,3]), np.random.choice([0,1,2,3])])
        else:
           self.agent_state = np.array([1, 1])

        self.info["Distance to goal"] = np.sqrt((self.agent_state[0]-self.goal[0])**2 +(self.agent_state[1]-self.goal[1])**2)

        return self.agent_state, self.info
    
    def add_obstacle_states(self, obstacle_state_coordinates):
        self.obstacle_states.append(np.array(obstacle_state_coordinates))


    def step(self, action):
        if action == 0 and self.agent_state[1] < self.grid_size -1: #- 1:  Right
            self.agent_state[1] += 1
        elif action == 1 and self.agent_state[1] > 0:  # Left
            self.agent_state[1] -= 1
        elif action == 2 and self.agent_state[0] > 0:  # Up
            self.agent_state[0] -= 1
        elif action == 3 and self.agent_state[0] < self.grid_size -1: #- 1: Down
            self.agent_state[0] += 1

        x, y = self.agent_state

        if True in [np.array_equal(self.agent_state, each_obstacle) for each_obstacle in self.obstacle_states]:
            self.reward = -10 # Penalty for hitting obstacle
        elif np.array_equal(self.agent_state, self.goal):
            if len(self.points) > 0:
                self.reward = -1  # Penalize reaching the goal without collecting all points
                print("Goal reached too early! Still points left.")
                self.done = False
            else:
                self.reward = +200
                self.done = True
                print("Goal reached with all rewards collected!")
        elif [int(x), int(y)] in self.points:
            self.reward = +50 # Reward for collecting a point
            self.points = []  # Remove the collected point
        else:
            self.reward = 0 #-0.01 
            self.done = False   

        self.info["Distance to goal"] = np.sqrt((self.agent_state[0]-self.goal[0])**2 +(self.agent_state[1]-self.goal[1])**2)
 

        return self.agent_state, self.reward, self.done, self.info

    def render(self):
        for event in pygame.event.get():
            if event == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.screen.fill((255, 255, 255))

        for x in range(0, self.screen_size, self.cell_size):
            pygame.draw.line(self.screen, (200, 200, 200), (x, 0), (x, self.screen_size))
        for y in range(0, self.screen_size, self.cell_size):
            pygame.draw.line(self.screen, (200, 200, 200), (0, y), (self.screen_size, y))

        for x, y in self.obstacle_states:
            self.screen.blit(self.obstacle_img, (y * self.cell_size, x * self.cell_size))

        for x, y in self.points:
            self.screen.blit(self.points_img, (y * self.cell_size, x * self.cell_size))

        goal_x, goal_y = self.goal
        self.screen.blit(self.goal_img, (goal_y * self.cell_size, goal_x * self.cell_size))

        agent_x, agent_y = self.agent_state
        self.screen.blit(self.agent_img, (agent_y * self.cell_size, agent_x * self.cell_size))

        pygame.draw.rect(self.screen, (0, 0, 0), (0, 0, self.screen_size, self.screen_size), 2)

        pygame.display.flip()
        self.clock.tick(10)

    def close(self):
        pygame.quit()


def create_env(goal_coordinates, obstacle_state_coordinates):#, random_initialization=False):
    Env = PadmEnvPygame(goal_coordinates=goal_coordinates)#, random_initialization=random_initialization)
    for i in range(len(obstacle_state_coordinates)):
        Env.add_obstacle_states(obstacle_state_coordinates=obstacle_state_coordinates[i])
    return Env
  