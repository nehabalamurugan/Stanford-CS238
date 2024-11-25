include("../src/problems/crying_baby.jl")
include("../src/core/exact_belief_state_planning.jl")
include("../src/core/exact_solution_methods.jl")

# Create the POMDP instance
pomdp = CryingBabyPOMDP(0.9)

# Initialize a belief state (uniform distribution over states)
# [P(sated), P(hungry)]
belief = [0.5, 0.5]

# Example 19.3: Update belief after taking action IGNORE and observing CRYING
# new_belief = update(belief, pomdp, IGNORE, CRYING)
# println("After IGNORE+CRYING: ", new_belief)


# Example 20.2: applying the lookahead to the crying baby problem
alpha_vectors = [[-3.7, -15], [-2, -21]]
policy = LookaheadAlphaVectorPolicy(pomdp, alpha_vectors)
optimal_action = policy(belief)
println("Optimal action: ", optimal_action)

# # computing the optimal value
# a = [[10, -4], [-1, 3], [-10, 3]]
# b = [0.6, 0.4]
# println("Optimal value function: ", optimal_valuefn(a, b))


# Exercise 7.5 policy evaluation using matrices
# Create a 4x4 matrix with some initial values
T = [
    0.3 0.7 0.0 0.0;
    0.0 0.85 0.15 0.0;
    0.0 0.0 0.0 1.0;
    0.0 0.0 0.0 1.0
]
R = [-0.3, -0.85, 10, 0]
U = compute_value_function(T, R, 0.9)
println("Value function: ", U)
