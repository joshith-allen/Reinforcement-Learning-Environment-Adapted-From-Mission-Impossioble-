import numpy as np
from Env import create_env
import time

# Load the trained Q-table
q_table_points = np.load("/home/josith/Documents/PADM/Reinforcement Learning (Environment Adapted From Mission Impossioble)/q_table_points.npy")
q_table_goal = np.load("/home/josith/Documents/PADM/Reinforcement Learning (Environment Adapted From Mission Impossioble)/q_table_goal.npy")

# Create environment
env = create_env(goal_coordinates=(3, 6), obstacle_state_coordinates=[(6, 5), (4, 3), (3, 1), (1, 5)])

# Reset environment
state, _ = env.reset()
state = tuple(state)

# Simulation control variables
done = False
key_collected = False

# Run the simulation
while not done:
    env.render()
    time.sleep(0.3)  # control speed of simulation

    if not key_collected:
        action = np.argmax(q_table_points[state])
    else:
        action = np.argmax(q_table_goal[state])

    next_state, reward, done, _ = env.step(action)
    next_state = tuple(next_state)

    if len(env.points) == 0 and not key_collected:
        key_collected = True
        print(" Point collected! Switching to goal Q-table")

    state = next_state

env.close()
print(" Simulation finished")
