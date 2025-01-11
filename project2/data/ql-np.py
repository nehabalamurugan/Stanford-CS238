import numpy as np
import pandas as pd
import random
from tqdm import trange  # For progress bar

# Q-learning parameters
learning_rate = 0.1
gamma = 0.95
n_training_episodes = 1000
max_steps = 100
min_epsilon = 0.01
max_epsilon = 1.0
decay_rate = 0.01

# Load CSV data
data = pd.read_csv("/Users/nehabalamurugan/Desktop/cs238/AA228-CS238-Student/project2/data/small.csv")

# Initialize Q-table (100 states, 4 actions)
num_states = data['s'].unique().size
num_actions = data['a'].unique().size
Qtable = np.zeros((num_states, num_actions))

def epsilon_greedy_policy(Qtable, state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.choice(range(num_actions))  # Explore
    else:
        return np.argmax(Qtable[state])  # Exploit

def get_sample(state, action):
    state -= 1
    possible_transitions = data[(data['s'] == state) & (data['a'] == action)]
    if not possible_transitions.empty:
        sample = possible_transitions.sample(1).iloc[0]
        next_state = int(sample['sp'])-1
        reward = sample['r']
        return next_state, reward 
    else:
        return state, 0  # No transition found; stay in the same state with zero reward

def train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, max_steps, Qtable):
    for episode in trange(n_training_episodes):
        # Decay epsilon over time
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        
        # Initialize starting state randomly
        state = random.choice(data['s'].unique())-1 #index value
        #print("state: ", state)
        
        for step in range(max_steps):
            #print("step: ", step)
            action = epsilon_greedy_policy(Qtable, state, epsilon)
    
            next_state, reward = get_sample(state, action)
            
            # Q update rule
            best_next_action = np.argmax(Qtable[next_state])
            td_target = reward + gamma * Qtable[next_state, best_next_action]
            Qtable[state, action] += learning_rate * (td_target - Qtable[state, action])
            
            state = next_state
            
    return Qtable

# Train the Q-table using the CSV data
trained_Qtable = train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, max_steps, Qtable)

# Extract the policy from the Q-table
policy = {state: np.argmax(trained_Qtable[state]) for state in range(num_states)}

# Write the policy to a file
with open("small.policy", "w") as f:
    for state, action in policy.items():
        f.write(f"{action + 1}\n")

print("Optimal policy saved to small.policy")
