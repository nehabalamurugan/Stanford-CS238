using JuMP, GLPK

# belief update: using bayes' rule to update the belief state   
function update(b::Vector{Float64}, 𝒫, a, o)
    𝒮, T, O = 𝒫.𝒮, 𝒫.T, 𝒫.O
    b′ = similar(b)
    for (i′, s′) in enumerate(𝒮)
        po = O(a, s′, o)
        b′[i′] = po * sum(T(s, a, s′) * b[i] for (i, s) in enumerate(𝒮))
    end
    if sum(b′) ≈ 0.0
        fill!(b′, 1)
    end
    return normalize!(b′, 1)
end

# lookahead for a POMDP 
function lookahead(𝒫::POMDP, U, s, a)
    𝒮, 𝒪, T, O, R, γ = 𝒫.𝒮, 𝒫.𝒪, 𝒫.T, 𝒫.O, 𝒫.R, 𝒫.γ
    u′ = sum(T(s, a, s′) * sum(O(a, s′, o) * U(o, s′) for o in 𝒪) for s′ in 𝒮)
    return R(s, a) + γ * u′
end

################
# conditional plan
################
struct ConditionalPlan
    a # action to take at root
    subplans # dictionary mapping observations to subplans
end

ConditionalPlan(a) = ConditionalPlan(a, Dict())
(π::ConditionalPlan)() = π.a
(π::ConditionalPlan)(o) = π.subplans[o]

function evaluate_plan(𝒫::POMDP, π::ConditionalPlan, s)
    U(o, s′) = evaluate_plan(𝒫, π(o), s′)
    return isempty(π.subplans) ? 𝒫.R(s, π()) : lookahead(𝒫, U, s, π())
end


###############
# alpha vectors
###############
function alphavector(𝒫::POMDP, π::ConditionalPlan)
    return [evaluate_plan(𝒫, π, s) for s in 𝒫.𝒮]
end
# A policy that selects actions based on the alpha vector
struct AlphaVectorPolicy
    𝒫 # POMDP problem
    Γ # alpha vectors
    a # actions associated with alpha vectors
end
function utility(π::AlphaVectorPolicy, b)
    return maximum(α ⋅ b for α in π.Γ)
end
function (π::AlphaVectorPolicy)(b)
    i = argmax([α ⋅ b for α in π.Γ])
    return π.a[i]
end

function optimal_valuefn(alpha, b)
    println("Dot products of alpha vectors with belief:")
    values = [α ⋅ b for α in alpha]
    for (i, val) in enumerate(values)
        println("α$i ⋅ b = $val")
    end
    return maximum(values)
end

###############
# one step lookahead
###############

function lookahead(𝒫::POMDP, U, b::Vector, a)
    # println("LOOKAHEAD VALUES--------------")
    # println("Action: ", a)
    𝒮, 𝒪, T, O, R, γ = 𝒫.𝒮, 𝒫.𝒪, 𝒫.T, 𝒫.O, 𝒫.R, 𝒫.γ
    r = sum(R(s, a) * b[i] for (i, s) in enumerate(𝒮))
    # println("Reward: ", r)

    Posa(o, s, a) = sum(O(a, s′, o) * T(s, a, s′) for s′ in 𝒮)
    Poba(o, b, a) = sum(b[i] * Posa(o, s, a) for (i, s) in enumerate(𝒮))

    total = 0.0
    for o in 𝒪
        prob = Poba(o, b, a)
        updated_belief = update(b, 𝒫, a, o)
        future_value = U(updated_belief)
        contribution = prob * future_value
        # println("Observation: ", o)
        # println("  P(o|b,a): ", prob)
        # println("  Updated belief: ", updated_belief)
        # println("  Future value: ", future_value)
        # println("  Contribution: ", γ * contribution)
        total += γ * contribution
    end
    # println("Total future reward: ", total)
    println("Total value: ", r + total)
    return r + total
end

function greedy(𝒫::POMDP, U, b::Vector)
    u, a = findmax(a -> lookahead(𝒫, U, b, a), 𝒫.𝒜)
    return (a=a, u=u)
end

struct LookaheadAlphaVectorPolicy
    𝒫 # POMDP problem
    Γ # alpha vectors
end

function utility(π::LookaheadAlphaVectorPolicy, b)
    val = maximum(α ⋅ b for α in π.Γ)
    println("At belief $b, utility is $val")
    return val
end
function greedy(π, b)
    U(b) = utility(π, b)
    return greedy(π.𝒫, U, b)
end
(π::LookaheadAlphaVectorPolicy)(b) = greedy(π, b).a #remove a to see the utility


################
# pruning alpha vectors
################    
function find_maximal_belief(α, Γ)
    m = length(α)
    if isempty(Γ)
        return fill(1 / m, m) # arbitrary belief
    end
    model = Model(GLPK.Optimizer)
    @variable(model, δ)
    @variable(model, b[i=1:m] ≥ 0)
    @constraint(model, sum(b) == 1.0)
    for a in Γ
        @constraint(model, (α - a) ⋅ b ≥ δ)
    end
    @objective(model, Max, δ)
    optimize!(model)
    return value(δ) > 0 ? value.(b) : nothing
end

function find_dominating(Γ)
    n = length(Γ)
    candidates, dominating = trues(n), falses(n)
    while any(candidates)
        i = findfirst(candidates)
        b = find_maximal_belief(Γ[i], Γ[dominating])
        if b === nothing
            candidates[i] = false
        else
            k = argmax([candidates[j] ? b ⋅ Γ[j] : -Inf for j in 1:n])
            candidates[k], dominating[k] = false, true
        end
    end
    return dominating
end
function prune(plans, Γ)
    d = find_dominating(Γ)
    return (plans[d], Γ[d])
end



################
# value iteration
################
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
function value_iteration(𝒫::POMDP, k_max)
    𝒮, 𝒜, R = 𝒫.𝒮, 𝒫.𝒜, 𝒫.R
    plans = [ConditionalPlan(a) for a in 𝒜]
    Γ = [[R(s, a) for s in 𝒮] for a in 𝒜]
    plans, Γ = prune(plans, Γ)
    for k in 2:k_max
        plans, Γ = expand(plans, Γ, 𝒫)
        plans, Γ = prune(plans, Γ)
    end
    return (plans, Γ)
end
function solve(M::ValueIteration, 𝒫::POMDP)
    plans, Γ = value_iteration(𝒫, M.k_max)
    return LookaheadAlphaVectorPolicy(𝒫, Γ)
end



#####
# 22: forward search 
#####   
struct ForwardSearch
    𝒫 # problem
    d # depth
    U # value function at depth d
end
function forward_search(𝒫, s, d, U)
    if d ≤ 0
        return (a=nothing, u=U(s))
    end
    best = (a=nothing, u=-Inf)
    U′(s) = forward_search(𝒫, s, d - 1, U).u
    for a in 𝒫.𝒜
        println("Action: $a at depth $d")
        u = lookahead(𝒫, U′, s, a)
        if u > best.u
            best = (a=a, u=u)
        end
    end
    return best
end
(π::ForwardSearch)(s) = forward_search(π.𝒫, s, π.d, π.U).a




export LookaheadAlphaVectorPolicy, AlphaVectorPolicy, utility, greedy, optimal_valuefn