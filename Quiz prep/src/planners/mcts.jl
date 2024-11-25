struct MonteCarloTreeSearch
    𝒫 # problem
    N # visit counts
    Q # action value estimates
    d # depth
    m # number of simulations
    c # exploration constant
    U # value function estimate
end

function (π::MonteCarloTreeSearch)(s)
    for k in 1:π.m
        simulate!(π, s)
    end
    return argmax(a -> π.Q[(s, a)], π.𝒫.𝒜)
end


function simulate!(π::MonteCarloTreeSearch, s, d=π.d)
    if d ≤ 0
        return π.U(s)
    end

    𝒫, N, Q, c = π.𝒫, π.N, π.Q, π.c
    𝒜, TR, γ = 𝒫.𝒜, 𝒫.TR, 𝒫.γ

    if !haskey(N, (s, first(𝒜)))
        for a in 𝒜
            N[(s, a)] = 0
            Q[(s, a)] = 0.0
        end
        return π.U(s)
    end
    a = explore(π, s)
    s′, r = TR(s, a)
    q = r + γ * simulate!(π, s′, d - 1)
    N[(s, a)] += 1
    Q[(s, a)] += (q - Q[(s, a)]) / N[(s, a)]
    return q
end


bonus(Nsa, Ns) = Nsa == 0 ? Inf : sqrt(log(Ns) / Nsa)

function explore(π::MonteCarloTreeSearch, s)
    𝒜, N, Q, c = π.𝒫.𝒜, π.N, π.Q, π.c
    Ns = sum(N[(s, a)] for a in 𝒜)
    return argmax(a -> Q[(s, a)] + c * bonus(N[(s, a)], Ns), 𝒜)
end