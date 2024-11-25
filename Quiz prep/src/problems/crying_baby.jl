using LinearAlgebra
include("../../src/types/mdp_types.jl")
using .models

# States: sated=1, hungry=2
const SATED = 1
const HUNGRY = 2
const STATES = [SATED, HUNGRY]

# Actions: feed=1, sing=2, ignore=3  
const FEED = 1
const SING = 2
const IGNORE = 3
const ACTIONS = [FEED, SING, IGNORE]

# Observations: crying=1, quiet=2
const CRYING = 1
const QUIET = 2
const OBSERVATIONS = [CRYING, QUIET]

# Transition function
function transition(s, a, s′)
    if s == HUNGRY && a == FEED
        return s′ == SATED ? 1.0 : 0.0
    elseif s == HUNGRY && a == SING
        return s′ == HUNGRY ? 1.0 : 0.0
    elseif s == HUNGRY && a == IGNORE
        return s′ == HUNGRY ? 1.0 : 0.0
    elseif s == SATED && a == FEED
        return s′ == SATED ? 1.0 : 0.0
    elseif s == SATED && a == SING
        return s′ == HUNGRY ? 0.1 : 0.9
    elseif s == SATED && a == IGNORE
        return s′ == HUNGRY ? 0.1 : 0.9
    end
end

# Observation function
function observation(a, s′, o)
    if s′ == HUNGRY
        if a == FEED
            return o == CRYING ? 0.8 : 0.2
        elseif a == SING
            return o == CRYING ? 0.9 : 0.1
        elseif a == IGNORE
            return o == CRYING ? 0.8 : 0.2
        end
    elseif s′ == SATED
        if a == FEED
            return o == CRYING ? 0.1 : 0.9
        elseif a == SING
            return o == CRYING ? 0.0 : 1.0
        elseif a == IGNORE
            return o == CRYING ? 0.1 : 0.9
        end
    end
end

# Reward function
function reward(s, a)
    r = s == HUNGRY ? -10.0 : 0.0
    if a == FEED
        r -= 5.0
    elseif a == SING
        r -= 0.5
    end
    return r
end

# Constructor
function CryingBabyPOMDP(γ::Float64)
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

# Export
export CryingBabyPOMDP, SATED, HUNGRY, FEED, SING, IGNORE, CRYING, QUIET, update