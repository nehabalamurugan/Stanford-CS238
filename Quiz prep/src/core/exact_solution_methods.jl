using LinearAlgebra

# MDP lookahead
function lookahead(𝒫::MDP, U, s, a)
    𝒮, T, R, γ = 𝒫.𝒮, 𝒫.T, 𝒫.R, 𝒫.γ
    return R(s, a) + γ * sum(T(s, a, s′) * U(s′) for s′ in 𝒮)
end

function lookahead(𝒫::MDP, U::Vector, s, a)
    𝒮, T, R, γ = 𝒫.𝒮, 𝒫.T, 𝒫.R, 𝒫.γ
    #println("Calculating lookahead for state: $s, action: $a")
    reward = R(s, a)
    #println("Immediate reward: $reward")
    future_value = γ * sum(T(s, a, s′) * U[i] for (i, s′) in enumerate(𝒮))
    #println("Future value: $future_value")
    return reward + future_value
end


################
# iterative policy evaluation: computes the utilility of a policy π 
################
function iterative_policy_evaluation(𝒫::MDP, π, k_max)
    𝒮, T, R, γ = 𝒫.𝒮, 𝒫.T, 𝒫.R, 𝒫.γ
    U = [0.0 for s in 𝒮]
    for k in 1:k_max
        U = [lookahead(𝒫, U, s, π(s)) for s in 𝒮]
    end
    return U
end

################
# policy evaluation: computes the utilility of a policy π 
################
function policy_evaluation(𝒫::MDP, π)
    𝒮, R, T, γ = 𝒫.𝒮, 𝒫.R, 𝒫.T, 𝒫.γ
    R′ = [R(s, π(s)) for s in 𝒮]
    T′ = [T(s, π(s), s′) for s in 𝒮, s′ in 𝒮]
    return (I - γ * T′) \ R′
end

# helper function to compute the value function
function compute_value_function(T::Matrix{Float64}, R::Vector{Float64}, γ::Float64)
    #I = Matrix{Float64}(I, size(T, 1), size(T, 2))
    return (I - γ * T) \ R
end

################
# value function policy: A value function policy extracted from a value function U 
#for an MDP 𝒫. The greedy function will be used in other algorithms.
################
struct ValueFunctionPolicy
    𝒫 # problem
    U # utility function
end

function greedy(𝒫::MDP, U, s)
    u, a = findmax(a -> lookahead(𝒫, U, s, a), 𝒫.𝒜)
    return (a=a, u=u)
end
(π::ValueFunctionPolicy)(s) = greedy(π.𝒫, π.U, s).a



################
# policy iteration: computes the optimal policy for an MDP 𝒫
################
struct PolicyIteration
    π # initial policy
    k_max # maximum number of iterations
end

function solve(M::PolicyIteration, 𝒫::MDP)
    π, 𝒮 = M.π, 𝒫.𝒮
    for k = 1:M.k_max
        U = policy_evaluation(𝒫, π)
        π′ = ValueFunctionPolicy(𝒫, U)
        if all(π(s) == π′(s) for s in 𝒮)
            break
        end
        π = π′
    end
    return π
end


################
# value iteration: computes the optimal policy for an MDP 𝒫
################
function backup(𝒫::MDP, U, s)
    return maximum(lookahead(𝒫, U, s, a) for a in 𝒫.𝒜)
end

struct ValueIteration
    k_max # maximum number of iterations
end

function solve(M::ValueIteration, 𝒫::MDP)
    U = [0.0 for s in 𝒫.𝒮]
    for k = 1:M.k_max
        U = [backup(𝒫, U, s) for s in 𝒫.𝒮]
    end
    return ValueFunctionPolicy(𝒫, U)
end

export ValueIteration, PolicyIteration, solve, ValueFunctionPolicy, iterative_policy_evaluation, policy_evaluation



