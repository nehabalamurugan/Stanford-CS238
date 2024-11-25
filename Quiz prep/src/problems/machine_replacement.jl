# Define states, actions, and observations
states = [0, 1, 2]  # Number of faulty components
actions = [:manufacture, :examine, :interrupt, :replace]
observations = [1, 2]  # 1 = non-defective, 2 = defective


T = Dict(
    :manufacture => [
        [0.81, 0.18, 0.01],
        [0.0, 0.9, 0.1],
        [0.0, 0.0, 1.0]
    ],
    :examine => [
        [0.81, 0.18, 0.01],
        [0.0, 0.9, 0.1],
        [0.0, 0.0, 1.0]
    ],
    :interrupt => [
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0]
    ],
    :replace => [
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0]
    ]
)

# Access transition probability
function T_function(s, a, sp)
    return T[a][s+1][sp+1]  # +1 to match array indices
end

O = Dict(
    :manufacture => [
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0]
    ],
    :examine => [
        [1.0, 0.0],
        [0.5, 0.5],
        [0.25, 0.75]
    ],
    :interrupt => [
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0]
    ],
    :replace => [
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0]
    ]
)

function O_function(a, sp, o)
    return O[a][sp+1][o]
end


R = Dict(
    :manufacture => [0.9025, 0.475, 0.25],
    :examine => [0.6525, 0.225, 0.0],
    :interrupt => [-0.5, -1.5, -2.5],
    :replace => [-2.0, -2.0, -2.0]
)

function R_function(s, a)
    return R[a][s+1]
end

function MachineReplacementPOMDP(γ::Float64)
    return POMDP(
        γ,
        states,
        actions,
        observations,
        T_function,
        O_function,
        R_function
    )
end

export MachineReplacementPOMDP
