# Imports:
# --------
from Env import create_env
from QLearn import train_q_learning, visualize_q_table

# User definitions:
# -----------------
train = True
visualize_results = True

"""
NOTE: Sometimes a fixed initializtion might push the agent to a local minimum.
In this case, it is better to use a random initialization.  
"""
#random_initialization = False  # If True, the Q-table will be initialized randomly

learning_rate = 0.01 # Learning rate #0.01
gamma = 0.99 # Discount factor #0.99
epsilon = 1.0 # Exploration rate #1.0
epsilon_min = 0.1 # Minimum exploration rate #0.1
epsilon_decay = 0.995 # Decay rate for exploration #0.995
no_episodes = 10000 #Number of episodes #10000


goal_coordinates = (3,6)  # Goal coordinates

points_coordinates = [(5, 1)]  # Points to collect



# Define all hell state coordinates as a tuple within a list
obstacle_state_coordinates = [(6, 5), (4, 3), (3, 1), (1, 5)]


# Execute:
# --------
if train:
    # Create an instance of the environment:
    # --------------------------------------
    env = create_env(goal_coordinates=(3,6),
                     obstacle_state_coordinates=[(6, 5), (4, 3), (3, 1), (1, 5)])#,random_initialization=random_initialization)

    # Train a Q-learning agent:
    # -------------------------
    train_q_learning(env=env,
                     no_episodes=no_episodes,
                     epsilon=epsilon,
                     epsilon_min=epsilon_min,
                     epsilon_decay=epsilon_decay,
                     alpha=learning_rate,
                     gamma=gamma)
    

if visualize_results:
    # Visualize the Q-table:
    # ----------------------
    visualize_q_table(obstacle_state_coordinates=obstacle_state_coordinates,
                      goal_coordinates=goal_coordinates,points=points_coordinates,
                      q_values_path="q_table_points.npy")
    
    visualize_q_table(obstacle_state_coordinates=obstacle_state_coordinates,
                      goal_coordinates=goal_coordinates,points=[],
                      q_values_path="q_table_goal.npy")
