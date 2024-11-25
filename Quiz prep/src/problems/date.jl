using LinearAlgebra
include("../../src/types/mdp_types.jl")
using .models

# States: interested=1, not_interested=2, ended=3
const INTERESTED = 1
const NOT_INTERESTED = 2
const ENDED = 3
const STATES = [INTERESTED, NOT_INTERESTED, ENDED]

# Actions: talk_about_ourselves=1, ask_about_them=2, end_date=3
const TALK = 1
const ASK = 2
const END = 3
const ACTIONS = [TALK, ASK, END]

# Observations: positive_response=1, negative_response=2, neutral_response=3
const POSITIVE = 1
const NEGATIVE = 2
const NEUTRAL = 3
const OBSERVATIONS = [POSITIVE, NEGATIVE, NEUTRAL]

# Transition function
function transition(s, a, s′)
    if s == INTERESTED
        if a == TALK
            return s′ == INTERESTED ? 0.7 : (s′ == NOT_INTERESTED ? 0.3 : 0.0)
        elseif a == ASK
            return s′ == INTERESTED ? 1.0 : 0.0
        elseif a == END
            return s′ == ENDED ? 1.0 : 0.0
        end
    elseif s == NOT_INTERESTED
        if a == TALK
            return s′ == NOT_INTERESTED ? 1.0 : 0.0
        elseif a == ASK
            return s′ == INTERESTED ? 0.5 : (s′ == NOT_INTERESTED ? 0.5 : 0.0)
        elseif a == END
            return s′ == ENDED ? 1.0 : 0.0
        end
    elseif s == ENDED
        return s′ == ENDED ? 1.0 : 0.0
    end
    return 0.0
end

# Reward function
function reward(s, a)
    if s == INTERESTED
        if a == TALK
            return 0.0
        elseif a == ASK
            return 1.1
        elseif a == END
            return 10.0
        end
    elseif s == NOT_INTERESTED
        if a == TALK
            return -1.0
        elseif a == ASK
            return 0.0
        elseif a == END
            return 0.0
        end
    elseif s == ENDED
        return 0.0
    end
    return 0.0
end

# Observation function
function observation(a, s′, o)
    if s′ == INTERESTED
        if o == POSITIVE
            return 0.7
        elseif o == NEGATIVE
            return 0.05
        elseif o == NEUTRAL
            return 0.25
        end
    elseif s′ == NOT_INTERESTED
        if o == POSITIVE
            return 0.05
        elseif o == NEGATIVE
            return 0.25
        elseif o == NEUTRAL
            return 0.7
        end
    elseif s′ == ENDED
        if o == NEUTRAL
            return 1.0
        else
            return 0.0
        end
    end
    return 0.0
end

function DatingMDP(γ::Float64)
    return MDP(
        γ,
        STATES,
        ACTIONS,
        transition,
        reward
    )
end

function DatingPOMDP(γ::Float64)
    return POMDP(
        γ,
        STATES,
        ACTIONS,
        OBSERVATIONS,
        transition,
        observation,
        reward
    )
end


export DatingMDP, DatingPOMDP, INTERESTED, NOT_INTERESTED, ENDED, TALK, ASK, END, POSITIVE, NEGATIVE, NEUTRAL, OBSERVATIONS, ACTIONS, STATES