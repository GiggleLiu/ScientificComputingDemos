export treeverse, treeverse_solve

struct TreeverseAction
    action::Symbol
    τ::Int
    δ::Int
    step::Int
    depth::Int
end

struct TreeverseLog
    actions::Vector{TreeverseAction}
    depth::Base.RefValue{Int}
    peak_mem::Base.RefValue{Int}  # should be `n*(k-1)+2`
end
TreeverseLog() = TreeverseLog(TreeverseAction[], Ref(0), Ref(0))
Base.push!(tlog::TreeverseLog, args...) = push!(tlog.actions, TreeverseAction(args..., tlog.depth[]))

Base.show(io::IO, ::MIME"text/plain", logger::TreeverseLog) = Base.show(io, logger)
function Base.show(io::IO, logger::TreeverseLog)
    print(io, """Treeverse log
| peak memory usage = $(logger.peak_mem[])
| number of function calls = $(count(x->x.action==:call, logger.actions))
| number of gradient calls = $(count(x->x.action==:grad, logger.actions))
| number of stack push/pop = $(count(x->x.action==:store, logger.actions))/$(count(x->x.action==:fetch, logger.actions))""")
end

function binomial_fit(N::Int, δ::Int)
    τ = 1
    while N > binomial(τ+δ, τ)
        τ += 1
    end
    return τ
end

function mid(δ, τ, σ, ϕ)
    κ = ceil(Int, (δ*σ + τ*ϕ)/(τ+δ))
    if κ >= ϕ && δ > 0
        κ = max(σ+1, ϕ-1)
    end
    return κ
end

"""
    treeverse(f, gf, s; δ, N, τ=binomial_fit(N,δ), f_inplace=true, logger = TreeverseLog())

Treeverse algorithm for back-propagating a program memory efficiently.

Positional arguments
* `f`, the step function that ``s_{i+1} = f(s_i)``,
* `gf`, the single step gradient function that ``g_i = gf(s_i, g_{i+1})``
   !!! When ``g_{i+1}`` is `nothing`, it should return the gradient of the loss function,
* `s`, the initial state ``s_0``,

Keyword arguments
* `δ`, the number of checkpoints,
* `N`, the number of time steps,
* `τ`, the number of sweeps, it is chosen as the smallest integer that `binomial(τ+δ, τ) >= N` by default,
* `f_inplace = false`, whether `f` is inplace,
* `logger = TreeverseLog()`, the logger.

Ref: https://www.tandfonline.com/doi/abs/10.1080/10556789208805505
"""
function treeverse(f, gf, s::T; δ, N, τ=binomial_fit(N,δ), f_inplace=false, logger = TreeverseLog()) where T
    state = Dict(0=>s)
    if N > binomial(τ+δ, τ)
        error("please input a larger `τ` and `δ` so that `binomial(τ+δ, τ) >= N`!")
    end
    g = treeverse!(f, gf, state, nothing, δ, τ, 0, 0, N, logger, f_inplace)
    return g
end

function treeverse!(f, gf, state::Dict{Int,T}, g, δ, τ, β, σ, ϕ, logger, f_inplace) where T
    logger.depth[] += 1
    # cache sσ
    if σ > β
        δ -= 1
        s = state[β]
        for j=β:σ-1
            s = getf(f, j)(s)
            push!(logger, :call, τ, δ, j)
        end
        store_state!(state,σ, f_inplace ? copy(s) : s)
        s = nothing
        push!(logger, :store, τ, δ, σ)
        logger.peak_mem[] = max(logger.peak_mem[], length(state))
    elseif σ < β
        error("treeverse fails! σ < β")
    end

    κ = mid(δ, τ, σ, ϕ)
    while τ>0 && κ < ϕ
        g = treeverse!(f, gf, state, g, δ, τ, σ, κ, ϕ, logger, f_inplace)
        τ -= 1
        ϕ = κ
        κ = mid(δ, τ, σ, ϕ)
    end

    g = getf(gf, σ)(state[σ], g)
    push!(logger, :grad, τ, δ, σ)
    if σ>β
        # remove state[σ]
        delete_state!(state, σ)
        push!(logger, :fetch, τ, δ, σ)
    end
    return g
end
getf(f, i) = f

@inline function store_state!(state::Dict, i::Int, x)
    state[i] = x
end

@inline function delete_state!(state::Dict, i::Int)
    pop!(state, i)
end

# Inputs:
# - `s`: The current seismic state containing wavefield values
# - `param`: Acoustic propagator parameters
# - `src`: Source location coordinates
# - `srcv`: Source time function values
# - `c`: Velocity model
# Returns the next seismic state after one time step
function treeverse_step(s, param, src, srcv, c)
    unext, φ, ψ = zero(s.u), copy(s.φ), copy(s.ψ)
    one_step!(param, unext, s.u, s.upre, φ, ψ, param.Σx, param.Σy, c)
    s2 = SeismicState(s.u, unext, φ, ψ, Ref(s.step[]+1))
    s2.u[src...] += srcv[s2.step[]]*param.DELTAT^2
    return s2
end

# Inputs:
# - `x`: The current seismic state
# - `g`: The gradient of the loss with respect to the next state
# - `param`: Acoustic propagator parameters
# - `src`: Source location coordinates
# - `srcv`: Source time function values
# - `gsrcv`: Gradient with respect to source time function
# - `c`: Velocity model
# - `gc`: Gradient with respect to velocity model
# Returns the gradients with respect to the current state, source time function, and velocity model
function treeverse_grad!(x, g, param, src, srcv, gsrcv, c, gc)
    # TODO: implement this with Enzyme.jl
    # one_step!(param, unext, s.u, s.upre, φ, ψ, param.Σx, param.Σy, c)
    Enzyme.autodiff(Reverse, treeverse_step, Const, Duplicated(x, g), Const(param), Const(src), Duplicated(srcv, gsrcv), Duplicated(c, gc))
    return (g, gsrcv, gc)
end

"""
    treeverse_solve(s0; param, src, srcv, c, δ=20, logger=TreeverseLog())

* `s0` is the initial state,
"""
function treeverse_solve(s0, gnf; param, src, srcv, c, δ=20, logger=TreeverseLog())
    f = x->treeverse_step(x, param, src, srcv, c)
    res = []
    function gf(x, g)
        if g === nothing
            y = f(x)
            push!(res, y)
            g = gnf(y)
        end
        treeverse_grad!(x, g[1], param, src, srcv, g[2], c, g[3])
        return g
    end
    g = treeverse(f, gf,
        copy(s0); δ=δ, N=param.NSTEP-1, f_inplace=false, logger=logger)
    return res[], g
end
