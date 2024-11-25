struct ForwardSearch <: OnlinePlanner
    problem::Problem
    depth::Int
    branch_factor::Int
    U::Function  # Add utility function field
end

function forward_search(𝒫, s, d, U)
    if d ≤ 0
        return (a=nothing, u=U(s))
    end

    best = (a=nothing, u=-Inf)

    U′(s) = forward_search(𝒫, s, d - 1, U).u

    for a in 𝒫.𝒜
        u = lookahead(𝒫, U′, s, a)
        if u > best.u
            best = (a=a, u=u)
        end
    end
    return best
end

(π::ForwardSearch)(s) = forward_search(π.problem, s, π.depth, π.U).a
