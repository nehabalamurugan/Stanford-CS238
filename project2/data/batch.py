import numpy as np
import pandas as pd
from tqdm import trange, tqdm 

# Q-learning parameters
learning_rate = 0.1
gamma = 0.95
epochs = 100  

data = pd.read_csv("/Users/nehabalamurugan/Desktop/cs238/AA228-CS238-Student/project2/data/large.csv")

# zero based indexing
data['s'] -= 1
data['sp'] -= 1
data['a'] -= 1

# Initialize Q-table
num_states = 302020
num_actions = 9
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
            Qtable[state, action] = C * np.sign(vel * acceleration)

#initialize_q_table(Qtable, C=1.0)

for epoch in trange(epochs, desc="Epochs"):
    for _, row in tqdm(data.iterrows(), total=len(data), desc="Samples", leave=False):
        state = int(row['s'])
        action = int(row['a'])
        reward = row['r']
        next_state = int(row['sp'])
        
        # Q-learning update rule
        best_next_action = np.argmax(Qtable[next_state])  # Choose best action in next state: is this needed?
        td_target = reward + gamma * Qtable[next_state, best_next_action]
        Qtable[state, action] += learning_rate * (td_target - Qtable[state, action])

print("Q-learning completed.")

# Extract the policy from the Q-table
policy = {state: np.argmax(Qtable[state]) + 1 for state in range(num_states)}  # Adjust back to 1-based actions

# Write the policy to a file
with open("large.policy", "w") as f:
    for state, action in policy.items():
        f.write(f"{action}\n")  

print("Optimal policy saved")
