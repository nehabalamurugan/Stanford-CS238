include("../src/problems/date.jl")
include("../src/core/exact_solution_methods.jl")
include("../src/core/exact_belief_state_planning.jl")
include("../src/core/offline_belief_state_planning.jl")

mdp = DatingMDP(0.9)

#question 1
# T_new = [0.0 0.0 1.0;
#         0.0 0.0 1.0;
#         0.0 0.0 1.0]
# R_vector = [10.0, 0.0, 0.0]

# question 2
# T_new = [0.7 0.3 0.0;
#     0.0 1.0 0.0;
#     0.0 0.0 1.0]
# R_vector = [0.0, -1.0, 0.0]

# question 3
# T_new = [1.0 0.0 0.0;
#     0.5 0.5 0.0;
#     0.0 0.0 1.0]
# R_vector = [1.1, 0.0, 0.0]

# U = compute_value_function(T_new, R_vector, 0.9)
# println("Value function: ", U)

# Example usage
γ = 0.9  # Discount factor
# mdp = DatingMDP(γ)
# π = s -> ASK  # Policy that always takes the END action
# util = policy_evaluation(mdp, π)  # Evaluate the utility of the policy
# println("Utility of the policy given ASK action for all states: ", util)

# #question 5
# val = ValueIteration(10000)
# sol = solve(val, mdp)
# println("Value iteration: ", sol.U)
# println("Value function policy: ", ValueFunctionPolicy(mdp, sol.U))

pomdp = DatingPOMDP(γ)
# question 8
# belief = [0.5, 0.5, 0.0]
# new_belief = update(belief, pomdp, TALK, NEUTRAL)
# println("New belief: ", new_belief)

#question 9
# belief = [1.0, 0.0, 0.0]
# t1 = update(belief, pomdp, TALK, POSITIVE)
# t2 = update(t1, pomdp, TALK, POSITIVE)
# t3 = update(t2, pomdp, TALK, POSITIVE)
# println("New belief: ", t3)


# question 12
# belief = [0.5, 0.5, 0.0]
# pol = LookaheadAlphaVectorPolicy(pomdp, [[0.0, -1.0, 0.0], [1.1, 0.0, 0.0], [10.0, 0.0, 0.0]])
# println("Utility: ", utility(pol, belief))


#question 15
#println("baws lowerbound: ", baws_lowerbound(pomdp))

#quesiton 16
#println("blind lowerbound: ", blind_lowerbound(pomdp, 2))


# question 17
belief = [0.4, 0.6, 0]
alpha_vectors = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
a = AlphaVectorPolicy(pomdp, alpha_vectors, ACTIONS)
U(belief) = utility(a, belief)

println("forward search: ", forward_search(pomdp, belief, 1, U))