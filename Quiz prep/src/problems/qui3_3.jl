include("../../src/types/mdp_types.jl")
using ..models

const Is = 1
const N = 2
const E = 3
const states = [Is, N, E]

const T = 1
const A = 2
const Ea = 3
const actions = [T, A, Ea]

function DatingMDP(γ::Float64)
    function reward(s, a)
        if s == :Is && a == :A
            return 1.1
        elseif s == :Is && a == :Ea
            return 10
        elseif s == :N && a == :T
            return -1.0
        end
        return 0.0
    end

    function transition(s, a, sp)
        if a == :Ea
            return sp == :E ? 1.0 : 0.0
        elseif s == :E
            return sp == :E ? 1.0 : 0.0
        elseif s == :Is && a == :T
            return sp == :Is ? 0.7 : sp == :N ? 0.3 : 0.0
        elseif s == :Is && a == :A
            return sp == :Is ? 1.0 : 0.0
        elseif s == :N && a == :A
            return sp == :Is ? 0.5 : sp == :N ? 0.5 : 0.0
        end
        return 0.0
    end


    return MDP(γ, states, actions, transition, reward)
end

export DatingMDP, Is, N, E, T, A, Ea
