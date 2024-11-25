include("exact_belief_state_planning.jl")

#QMDP
struct QMDP
    k_max # maximum number of iterations
end
function alphavector_iteration(𝒫::POMDP, M, Γ)
    for k in 1:M.k_max
        Γ = update(𝒫, M, Γ)
    end
    return Γ
end
function update(𝒫::POMDP, M::QMDP, Γ)
    𝒮, 𝒜, R, T, γ = 𝒫.𝒮, 𝒫.𝒜, 𝒫.R, 𝒫.T, 𝒫.γ
    Γ′ = [[R(s, a) + γ * sum(T(s, a, s′) * maximum(α′[j] for α′ in Γ)
                             for (j, s′) in enumerate(𝒮)) for s in 𝒮] for a in 𝒜]
    return Γ′
end
function solve(M::QMDP, 𝒫::POMDP)
    Γ = [zeros(length(𝒫.𝒮)) for a in 𝒫.𝒜]
    Γ = alphavector_iteration(𝒫, M, Γ)
    return AlphaVectorPolicy(𝒫, Γ, 𝒫.𝒜)
end


#Fast informed bound
struct FastInformedBound
    k_max # maximum number of iterations
end
function update(𝒫::POMDP, M::FastInformedBound, Γ)
    𝒮, 𝒜, 𝒪, R, T, O, γ = 𝒫.𝒮, 𝒫.𝒜, 𝒫.𝒪, 𝒫.R, 𝒫.T, 𝒫.O, 𝒫.γ
    Γ′ = [[R(s, a) + γ * sum(maximum(sum(O(a, s′, o) * T(s, a, s′) * α′[j]
                                         for (j, s′) in enumerate(𝒮)) for α′ in Γ) for o in 𝒪)
           for s in 𝒮] for a in 𝒜]
    return Γ′
end
function solve(M::FastInformedBound, 𝒫::POMDP)
    Γ = [zeros(length(𝒫.𝒮)) for a in 𝒫.𝒜]
    Γ = alphavector_iteration(𝒫, M, Γ)
    return AlphaVectorPolicy(𝒫, Γ, 𝒫.𝒜)
end


#Fast lower bound
function baws_lowerbound(𝒫::POMDP)
    # 𝒮, 𝒜, R, γ = 𝒫.𝒮, 𝒫.𝒜, 𝒫.R, 𝒫.γ
    # r = maximum(minimum(R(s, a) for s in 𝒮) for a in 𝒜) / (1 - γ)
    # α = fill(r, length(𝒮))
    α = [-10.0, -10.0, -10.0]
    return α
end


#Blind lower bound
function blind_lowerbound(𝒫, k_max)
    𝒮, 𝒜, T, R, γ = 𝒫.𝒮, 𝒫.𝒜, 𝒫.T, 𝒫.R, 𝒫.γ
    Q(s, a, α) = R(s, a) + γ * sum(T(s, a, s′) * α[j] for (j, s′) in enumerate(𝒮))
    Γ = [baws_lowerbound(𝒫) for a in 𝒜]
    for k in 1:k_max
        Γ = [[Q(s, a, α) for s in 𝒮] for (α, a) in zip(Γ, 𝒜)]
    end
    return Γ
end

#Point based value iteration
function backup(𝒫::POMDP, Γ, b)
    𝒮, 𝒜, 𝒪, γ = 𝒫.𝒮, 𝒫.𝒜, 𝒫.𝒪, 𝒫.γ
    R, T, O = 𝒫.R, 𝒫.T, 𝒫.O
    Γa = []
    for a in 𝒜
        Γao = []
        for o in 𝒪
            b′ = update(b, 𝒫, a, o)
            push!(Γao, argmax(α -> α ⋅ b′, Γ))
        end
        α = [R(s, a) + γ * sum(sum(T(s, a, s′) * O(a, s′, o) * Γao[i][j]
                                   for (j, s′) in enumerate(𝒮)) for (i, o) in enumerate(𝒪))
             for s in 𝒮]
        push!(Γa, α)
    end
    return argmax(α -> α ⋅ b, Γa)
end

struct PointBasedValueIteration
    B # set of belief points
    k_max # maximum number of iterations
end

function update(𝒫::POMDP, M::PointBasedValueIteration, Γ)
    return [backup(𝒫, Γ, b) for b in M.B]
end

function solve(M::PointBasedValueIteration, 𝒫)
    Γ = fill(baws_lowerbound(𝒫), length(𝒫.𝒜))
    Γ = alphavector_iteration(𝒫, M, Γ)
    return LookaheadAlphaVectorPolicy(𝒫, Γ)
end


# randomized point based value iteration    
struct RandomizedPointBasedValueIteration
    B # set of belief points
    k_max # maximum number of iterations
end
function update(𝒫::POMDP, M::RandomizedPointBasedValueIteration, Γ)
    Γ′, B′ = [], copy(M.B)
    while !isempty(B′)
        b = rand(B′)
        α = argmax(α -> α ⋅ b, Γ)
        α′ = backup(𝒫, Γ, b)
        if α′ ⋅ b ≥ α ⋅ b
            push!(Γ′, α′)
        else
            push!(Γ′, α)
        end
        filter!(b -> maximum(α ⋅ b for α in Γ′) <
                     maximum(α ⋅ b for α in Γ), B′)
    end
    return Γ′
end
function solve(M::RandomizedPointBasedValueIteration, 𝒫)
    Γ = [baws_lowerbound(𝒫)]
    Γ = alphavector_iteration(𝒫, M, Γ)
    return LookaheadAlphaVectorPolicy(𝒫, Γ)
end


export QMDP, FastInformedBound, baws_lowerbound, PointBasedValueIteration, RandomizedPointBasedValueIteration, solve