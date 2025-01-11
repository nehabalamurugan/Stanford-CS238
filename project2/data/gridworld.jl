using CSV
using DataFrames

# Configuration for MDP
grid_size = (10, 10)
γ = 0.95
k_max = 100

# Load CSV file into a DataFrame
data = CSV.read("/Users/nehabalamurugan/Desktop/cs238/AA228-CS238-Student/project2/data/medium.csv", DataFrame)

# Define state and action spaces
lin_indices = LinearIndices(grid_size)
num_states = length(lin_indices)
states = collect(1:num_states)
actions = unique(data.a)

# Initialize transition and reward dictionaries
transitions = Dict{Tuple{Int,Int},Dict{Int,Float64}}()
rewards = Dict{Tuple{Int,Int},Float64}()

# Step 1: Count occurrences of each (s, a, sp) to calculate probabilities
transition_counts = Dict{Tuple{Int,Int,Int},Int}()
action_counts = Dict{Tuple{Int,Int},Int}()

for row in eachrow(data)
    s, a, r, sp = row.s, row.a, row.r, row.sp

    # Update transition counts
    key = (s, a, sp)
    transition_counts[key] = get(transition_counts, key, 0) + 1

    # Update action counts
    action_key = (s, a)
    action_counts[action_key] = get(action_counts, action_key, 0) + 1

    # Update rewards for averaging later
    rewards[action_key] = get(rewards, action_key, 0.0) + r
end

# Step 2: Calculate probabilities for each transition
for ((s, a, sp), count) in transition_counts
    action_key = (s, a)
    if !haskey(transitions, action_key)
        transitions[action_key] = Dict{Int,Float64}()
    end
    # Probability of transitioning to sp given (s, a)
    transitions[action_key][sp] = count / action_counts[action_key]
end

# Step 3: Average rewards for each (s, a)
for (key, total_reward) in rewards
    rewards[key] = total_reward / action_counts[key]
end

# Step 4: Define transition and reward functions based on these dictionaries
function T(s, a, sp)
    return get(get(transitions, (s, a), Dict()), sp, 0.0)
end

function R(s, a)
    return get(rewards, (s, a), 0.0)
end

# Define the MDP struct
struct MDP
    𝒮  # States
    𝒜  # Actions
    T  # Transition function
    R  # Reward function
    γ  # Discount factor
end

# Create an MDP instance
mdp = MDP(states, actions, T, R, γ)

# Define Value Iteration and Policy structures
struct ValueIteration
    k_max
end

struct ValueFunctionPolicy
    𝒫  # MDP problem
    U  # Utility function
end

# Function to get the optimal action for each state from the policy
function get_optimal_action(policy::ValueFunctionPolicy, s)
    𝒫, U = policy.𝒫, policy.U
    return argmax([lookahead(𝒫, U, s, a) for a in 𝒫.𝒜])  # Return action with max utility
end

# Value Iteration algorithm
function solve(M::ValueIteration, 𝒫::MDP)
    U = [0.0 for s in 𝒫.𝒮]  # Initialize utility values to zero
    for k in 1:M.k_max
        U = [backup(𝒫, U, s) for s in 𝒫.𝒮]  # Update U for each state
    end
    return ValueFunctionPolicy(𝒫, U)  # Return an instance of ValueFunctionPolicy
end

# Backup function that uses T and R correctly
function backup(𝒫::MDP, U, s)
    return maximum(lookahead(𝒫, U, s, a) for a in 𝒫.𝒜)
end

# Lookahead function that uses T and R to calculate the expected value
function lookahead(𝒫::MDP, U, s, a)
    𝒮, T, R, γ = 𝒫.𝒮, 𝒫.T, 𝒫.R, 𝒫.γ
    # Calculate expected utility for taking action a in state s
    expected_value = sum(T(s, a, sp) * U[sp] for sp in 𝒮)
    return R(s, a) + γ * expected_value
end

# Save the policy to a file
function save_policy(policy::ValueFunctionPolicy, filename::String)
    open(filename, "w") do f
        for s in policy.𝒫.𝒮
            a = get_optimal_action(policy, s)
            println(f, "$s,$a")  # Save each state and its optimal action
        end
    end
end

# Run Value Iteration and save the policy
vi = ValueIteration(k_max)
policy = solve(vi, mdp)
save_policy(policy, "medium.policy")
println("Optimal policy saved to medium.policy")
