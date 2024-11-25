module models

export MDP, POMDP

struct MDP
    γ::Float64 # discount factor
    𝒮::Vector{Int}  # state space
    𝒜::Vector{Int}  # action space
    T::Function     # transition function
    R::Function     # reward function
end

struct POMDP
    γ::Float64 # discount factor
    𝒮::Vector{Int}  # state space
    𝒜::Vector{Int}  # action space
    𝒪::Vector{Int}  # observation space
    T::Function     # transition function
    O::Function     # observation function
    R::Function     # reward function
end

end