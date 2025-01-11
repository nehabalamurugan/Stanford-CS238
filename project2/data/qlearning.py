import numpy as np
import pandas as pd
import random

alpha = 0.1
gamma = 0.95
epsilon = 0.1
episodes = 1
max_steps = 100   

data = pd.read_csv("/Users/nehabalamurugan/Desktop/cs238/AA228-CS238-Student/project2/data/small.csv")

states = data['s'].unique()
actions = data['a'].unique()

#initialize q table: dictionary of dictionaries (state, action)
Q = {state: {action: 0 for action in actions} for state in states}
print(Q)
Qtable = np.zeros((states.size, actions.size))
print(Qtable)

#chose an action 
def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.choice(list(Q[state].keys())) #explore 
    else: 
        return max(Q[state], key=Q[state].get) #exploit
    
def get_reward(state, action, next_state):
    row = data[(data['s'] == state) & (data['a'] == action) & (data['sp'] == next_state)]
    if not row.empty:
        return row['r'].values[0]
    else: 
        return 0

def get_next_state(state, action):
    possible_transitions = data[(data['s']==state) & (data['a'] == action)]
    if not possible_transitions.empty:
        next_state = possible_transitions.sample(1, weights=possible_transitions['sp']).iloc[0]['sp']
        return next_state
    else: 
        return state

for episode in range(episodes):
    print("In episode: ", episode)
    state = random.choice(states)

    for step in range(max_steps):
        action = choose_action(state)
        next_state = get_next_state(state, action)
        reward = get_reward(state, action, next_state)
        
        # Update Q-value using the Q-learning formula
        best_next_action = max(Q[next_state], key=Q[next_state].get)
        td_target = reward + gamma * Q[next_state][best_next_action]
        td_delta = td_target - Q[state][action]
        Q[state][action] += alpha * td_delta
        
        # Update state
        state = next_state
        

policy = {state: max(actions, key=lambda action: Q[state][action]) for state in Q}

print("Optimal policy:")
with open("small.policy", "w") as f:
    for state, action in policy.items():
        f.write(f"{action}\n")