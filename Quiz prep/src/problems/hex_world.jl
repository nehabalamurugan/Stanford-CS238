include("../../src/types/mdp_types.jl")
using ..models

# Define the hex directions (clockwise from top)
const HEX_DIRECTIONS = [
    1,  # right
    2,  # down-right  
    3,  # down-left
    4,  # left
    5,  # up-left
    6   # up-right
]

function hex_world_straight_line(n)
    # States: 1 to n are valid cells, n+1 is absorbing state
    𝒮 = collect(1:(n+1))
    # Actions: 1-6 corresponding to hex directions
    𝒜 = collect(1:6)

    # Transition function
    function T(s, a, s′)
        # In absorbing state, stay there
        if s == n + 1
            return s′ == n + 1 ? 1.0 : 0.0

            # If at last state, go to absorbing state at any action
        elseif s == n
            return s′ == n + 1 ? 1.0 : 0.0

        elseif s == 1
            if a == 1
                return s′ == 2 ? 0.7 : s′ == s ? 0.3 : 0.0
            elseif a == 2 || a == 6
                return s′ == 2 ? 0.18 : s′ == s ? 0.85 : 0.0
            elseif a == 3 || a == 4 || a == 5
                return s′ == s ? 1.0 : 0.0
            end
        elseif s == 2
            if a == 1
                return s′ == 3 ? 0.7 : s′ == s ? 0.3 : 0.0
            elseif a == 2
                return s′ == 3 ? 0.15 : s′ == s ? 0.85 : 0.0
            elseif a == 3
                return s′ == 1 ? 0.15 : s′ == s ? 0.85 : 0.0
            elseif a == 4
                return s′ == 1 ? 0.7 : s′ == s ? 0.3 : 0.0
            elseif a == 5
                return s′ == 1 ? 0.15 : s′ == s ? 0.85 : 0.0
            elseif a == 6
                return s′ == 3 ? 0.15 : s′ == s ? 0.85 : 0.0
            end
        end
    end

    # Reward function
    function Rsas(s, a, s′)
        if s == n
            return 10.0
            #if hitting the border, return -1
        elseif s == n + 1
            return 0.0
        elseif s == s′
            return -1.0
        end
        return 0.0
    end

    function R(s, a)
        reward = 0.0
        for s′ in 𝒮
            reward += T(s, a, s′) * Rsas(s, a, s′)
        end
        return reward
    end

    return MDP(0.90, 𝒮, 𝒜, T, R)
end

export hex_world_straight_line