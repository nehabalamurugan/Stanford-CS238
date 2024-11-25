include("../src/problems/crying_baby.jl")
include("../src/core/exact_belief_state_planning.jl")

using .models

𝒫 = CryingBabyPOMDP(0.9)
# Initialize the belief state
belief = [0.5, 0.5]

# Define the alpha vectors
alpha_vectors = [[-3.7, -15], [-2, -21]]

# Create the value function using the alpha vectors
a = AlphaVectorPolicy(𝒫, alpha_vectors, ACTIONS)
U(belief) = utility(a, belief)
println("Value function: ", U(belief))

# Perform forward search to depth 2
depth = 2
optimal_action = forward_search(𝒫, belief, depth, U)

# Print the optimal action determined by the forward search
println("Optimal action from forward search: ", optimal_action)
