# Imports:
# --------
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Function 1: Train Q-learning agent
# -----------
def train_q_learning(env,no_episodes,
                     epsilon,epsilon_min,
                     epsilon_decay,
                     alpha,gamma):

    # Initialize the Q-table:
    # -----------------------
    q_table_points = np.zeros((env.grid_size, env.grid_size, env.action_space.n))
    q_table_goal = np.zeros((env.grid_size, env.grid_size, env.action_space.n))

    count = 0
    
    # Q-learning algorithm:
    # ---------------------
    #! Step 1: Run the algorithm for fixed number of episodes
    #! -------
    for episode in range(no_episodes):
        agent_state, _  = env.reset() # Agent State & Info

        count_render = no_episodes - 50
        count += 1

        agent_state = tuple(agent_state)
        total_reward = 0
        done = False
        key = False

        #! Step 2: Take actions in the environment until "Done" flag is triggered
        #! -------
        step_count = 0

        # while not done:
        for _ in range(1000):
            #! Step 3: Define your Exploration vs. Exploitation
            #! -------

            # Choose which Q-table to use
            # current_q_table = q_table_goal if key else q_table_points

            step_count += 1

            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                if key==False:
                    action = np.argmax(q_table_points[agent_state])  # Exploit action = np.argmax(current_q_table[agent_state[0], agent_state[1]])
                else:
                    action = np.argmax(q_table_goal[agent_state])

            next_state, reward, done, _ = env.step(action)

            if count >= count_render:
                env.render() # Opening the Render last few Steps
            
            #env.render()

            next_state = tuple(next_state)
            total_reward += reward

            # Switch to goal-reaching phase after all points are collected
            #! Step 4: Update the Q-values using the Q-value update rule
            #! -------

            #! Step 5: Update the Q-table based on the situation
            #! -------
            if key==False:
                # Update Q-table for collecting points
                q_table_points[agent_state][action] = q_table_points[agent_state][action] + alpha * (
                    reward + gamma * np.max(q_table_points[next_state]) - q_table_points[agent_state][action])
                if len(env.points) == 0:
                    key = True
                    print(f"Episode {episode+1}: All points collected. Switching to goal Q-table.")
            elif key==True:
                # Update Q-table for reaching the goal
                q_table_goal[agent_state][action] = q_table_goal[agent_state][action] + alpha * (
                        reward + gamma * np.max(q_table_goal[next_state]) - q_table_goal[agent_state][action])
                  
            agent_state = next_state
            if done:
                break
            

        #! Step 6: Perform epsilon decay
        #! -------
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        print(f"Episode {episode + 1}: Total Reward: {total_reward}: Steps: {step_count}")

    #! Step 7: Close the environment window
    #! -------
    env.close()
    print("Training finished.\n")

    #! Step 8: Save the trained Q-table
    # Save both Q-tables
    np.save("/home/josith/Documents/PADM/Reinforcement Learning (Environment Adapted From Mission Impossioble)/q_table_points.npy", q_table_points)
    np.save("/home/josith/Documents/PADM/Reinforcement Learning (Environment Adapted From Mission Impossioble)/q_table_goal.npy", q_table_goal)
    print("Saved both Q-tables.") 


# Function 2: Visualize the Q-table
# -----------
def visualize_q_table(obstacle_state_coordinates=[(6, 5), (4, 3), (3, 1), (1, 5)],
                      goal_coordinates=(6, 3),
                      points = [(1,5)],
                      actions=["Right", "Left", "Up", "Down"],
                      q_values_path="/home/josith/Documents/PADM/Reinforcement Learning (Environment Adapted From Mission Impossioble)/q_table_points.npy"):

    # Load the Q-table:
    # -----------------
    try:
        q_table = np.load(q_values_path)

        # Create subplots for each action:
        # --------------------------------
        _, axes = plt.subplots(1, 4, figsize=(20, 5))

        for i, action in enumerate(actions):
            ax = axes[i]
            heatmap_data = q_table[:, :, i].copy()

            # Mask the goal state's Q-value for visualization:
            # ------------------------------------------------
            mask = np.zeros_like(heatmap_data, dtype=bool)
            gx, gy = goal_coordinates
            initial_agent_coordinates = (1,1)
            Ax, Ay = initial_agent_coordinates
            mask[gx, gy] = True

            for hx, hy in obstacle_state_coordinates:
                mask[hx, hy] = True


            for ptx, pty in points:
                mask[ptx, pty] = True

            sns.heatmap(heatmap_data,annot=True,fmt=".2f",cmap="viridis",
                ax=ax,cbar=False,mask=mask,annot_kws={"size": 9},square=True,
                linewidths=0.1,linecolor='gray')#,vmin=np.min(q_table), vmax=np.max(q_table))

            # Denote Goal and Hell states:
            # ----------------------------
            
            #ax.invert_xaxis()  # This makes (0, 0) at top-left to match Pygame grid

            ax.set_title(f'Action: {action}')

            # Plot 'G' for goal
            ax.text(gy + 0.5, gx + 0.5, 'G', color='green',
                    ha='center', va='center', weight='bold', fontsize=14)
            
            ax.text(Ay + 0.5, Ax + 0.5, 'A', color='grey',
                    ha='center', va='center', weight='bold', fontsize=14)
            
            # Plot 'H' for all obstacle states
            for hx, hy in obstacle_state_coordinates:
                ax.text(hy + 0.5, hx + 0.5, 'H', color='red',
                        ha='center', va='center', weight='bold', fontsize=14)
            
            #Plot 'pt' for point states
            for ptx, pty in points:
                ax.text(pty + 0.5, ptx + 0.5, 'P', color='blue',
                        ha='center', va='center', weight='bold', fontsize=14)

        #plt.suptitle(f"Q-Table Visualization from: {q_values_path}", fontsize=16)
        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print("No saved Q-table was found. Please train the Q-learning agent first or check your path.")
