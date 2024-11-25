include("../src/problems/hex_world.jl")
include("../src/core/exact_solution_methods.jl")


# Example 7.10 policy iteration 
hex = hex_world_straight_line(3)
π_vector = [1, 6, 3, 1]  # Initial policy: east, northeast, southwest
π = s -> s == length(hex.𝒮) ? 1 : π_vector[findfirst(isequal(s), collect(hex.𝒮))]

# Print transition matrix
println("\nForming the transition matrix T:")
T_matrix = [hex.T(s, π(s), s′) for s in hex.𝒮, s′ in hex.𝒮]
display(T_matrix)

# Print reward vector for the policy
println("\nReward vector R:")
R_vector = [hex.R(s, π(s)) for s in hex.𝒮]
display(R_vector)

# Calculate and display value function
γ = hex.γ
I_matrix = Matrix{Float64}(I, length(hex.𝒮), length(hex.𝒮))
U = (I_matrix - γ * T_matrix) \ R_vector

display(U)

# Policy improvement step
println("\nPolicy improvement step:")
for s in hex.𝒮
    values = [lookahead(hex, U, s, a) for a in hex.𝒜]
    println("\nπ(s$s) = arg max$(values) = a$(argmax(values))")
end


# value iteration 
val = ValueIteration(2)
sol = solve(val, hex)
println(sol.U)