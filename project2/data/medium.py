import numpy as np
import pandas as pd
from tqdm import trange, tqdm  # Import tqdm for progress bars

# Q-learning parameters
learning_rate = 0.01
gamma = 0.95
lambda_smooth = 0.01  # Weight for the smoothness prior
epochs = 100  # Number of epochs for Q-learning

# Load and preprocess the dataset
data = pd.read_csv("/Users/nehabalamurugan/Desktop/cs238/AA228-CS238-Student/project2/data/medium.csv")

# Adjust for zero-based indexing in states and actions
data['s'] -= 1
data['sp'] -= 1
data['a'] -= 1

# Initialize Q-table (50,000 states, 7 actions)
num_states = 50000
num_actions = 7
Qtable = np.zeros((num_states, num_actions))

# Helper function to extract position and velocity from state
def extract_pos_vel(state, num_pos=500):
    pos = state % num_pos
    vel = state // num_pos
    return pos, vel

# Initialize Q-table with a directional bias
def initialize_q_table(Qtable, C=1.0):
    for state in range(num_states):
        pos, vel = extract_pos_vel(state)
        for action in range(num_actions):
            acceleration = action - 3  # Actions map to [-3, -2, ..., 3]
            Qtable[state, action] = C * np.sign(vel) * np.sign(acceleration)

# Identify terminal states and set their Q-values
def set_terminal_states(Qtable, data, max_reward):
    terminal_states = data[data['r'] == max_reward]['sp'].unique()
    for state in terminal_states:
        Qtable[state, :] = max_reward  # Set all actions to max_reward
    return terminal_states

# Modified reward function
def modified_reward(pos, vel, base_reward, pos_goal=499, pos_min=0, alpha=0.1, beta=0.05):
    velocity_reward = max(0, vel)  # Encourage positive velocity
    position_reward = (pos - pos_min) / (pos_goal - pos_min)  # Normalize position reward
    return base_reward + alpha * velocity_reward + beta * position_reward

# Update dataset with modified rewards
def update_rewards(data, pos_goal=499, pos_min=0, alpha=0.1, beta=0.05):
    for i, row in data.iterrows():
        state = int(row['s'])
        base_reward = row['r']
        pos, vel = extract_pos_vel(state)
        data.at[i, 'r'] = modified_reward(pos, vel, base_reward, pos_goal, pos_min, alpha, beta)

# Helper function to find neighboring states
def find_neighbors(state, num_pos=500, num_vel=100):
    pos, vel = extract_pos_vel(state, num_pos)
    neighbors = []
    for dp, dv in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1)]:
        new_pos, new_vel = pos + dp, vel + dv
        if 0 <= new_pos < num_pos and 0 <= new_vel < num_vel:  # Check bounds
            neighbor_state = new_pos + num_pos * new_vel
            neighbors.append(neighbor_state)
    return neighbors

# Initialize Q-table with bias
initialize_q_table(Qtable, C=1.0)

# Process terminal states
max_reward = data['r'].max()
terminal_states = set_terminal_states(Qtable, data, max_reward)

# Update dataset with the new reward function
update_rewards(data, pos_goal=499, pos_min=0, alpha=0.1, beta=0.05)

print(Qtable)

# Q-learning with smoothness prior
for epoch in trange(epochs, desc="Epochs"):
    for _, row in tqdm(data.iterrows(), total=len(data), desc="Samples", leave=False):
        state, action, reward, next_state = int(row['s']), int(row['a']), row['r'], int(row['sp'])

        # Q-learning update
        best_next_action = np.argmax(Qtable[next_state])  # Best action at next state
        td_target = reward + gamma * Qtable[next_state, best_next_action]
        td_error = td_target - Qtable[state, action]

        # Smoothness prior
        neighbors = find_neighbors(state)
        smoothness_loss = sum(np.sum((Qtable[state] - Qtable[neighbor]) ** 2) for neighbor in neighbors)
        smoothness_loss /= len(neighbors) if neighbors else 1  # Avoid division by zero

        # Update Q-value with smoothness loss
        Qtable[state, action] += learning_rate * (td_error - lambda_smooth * smoothness_loss)

print("Q-learning completed with smoothness prior.")

# Extract the optimal policy
policy = {state: np.argmax(Qtable[state]) + 1 for state in range(num_states)}  # Adjust back to 1-based actions

# Save the policy to a file
with open("medium.policy", "w") as f:
    for state, action in policy.items():
        f.write(f"{action}\n")

print("Optimal policy saved.")
