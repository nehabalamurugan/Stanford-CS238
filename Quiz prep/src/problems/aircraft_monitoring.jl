using LinearAlgebra

struct POMDP
    γ::Float64
    𝒮::Vector{Int}
    𝒜::Vector{Int}
    𝒪::Vector{Int}
    T::Function
    O::Function
    R::Function
end

#states: s
const NORMAL = 0
const MALFUNCTION = 1
const STATES = [NORMAL, MALFUNCTION]

#actions: m
const FLY = 0
const MAINTENANCE = 1
const ACTIONS = [FLY, MAINTENANCE]

#observations: w
const OK = 0
const WARNING = 1
const OBSERVATIONS = [WARNING, OK]

function transition(s, a, s′)
    if a == FLY && s == NORMAL
        return s′ == NORMAL ? 0.95 : 0.05
    elseif a == MAINTENANCE && s == NORMAL
        return s′ == NORMAL ? 1.0 : 0.0
    elseif a == FLY && s == MALFUNCTION
        return s′ == MALFUNCTION ? 1.0 : 0.0
    elseif a == MAINTENANCE && s == MALFUNCTION
        return s′ == NORMAL ? 0.98 : 0.02
    end
end

function observation(a, s′, o)
    if s′ == OK
        return o == OK ? 0.99 : 0.05
    elseif s′ == MALFUNCTION
        return o == WARNING ? 0.7 : 0.3
    end
end

function reward(s, a)
    r = s == HUNGRY ? -10.0 : 0.0
    if a == FEED
        r -= 5.0
    elseif a == SING
        r -= 0.5
    end
    return r
end

function AircraftMonitoringPOMDP(γ::Float64)
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

export AircraftMonitoringPOMDP, NORMAL, MALFUNCTION, OK, WARNING, FLY, MAINTENANCE
