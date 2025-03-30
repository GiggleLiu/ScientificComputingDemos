struct SimulatedBifurcation{T, KIND}
    a::T
    c0::T
    g::SimpleGraph{Int}
    J::Vector{T}
    function SimulatedBifurcation{KIND}(a::T, c0::T, g::SimpleGraph{Int}, J::Vector{T}) where {T, KIND}
        @assert KIND in (:aSB, :bSB, :dSB) "Invalid bifurcation type: $KIND, must be one of (:aSB, :bSB, :dSB)"
        @assert length(J) == ne(g) "Length of J must be equal to the number of edges in the graph, got $(length(J)) and $(ne(g))"
        return new{T, KIND}(a, c0, g, J)
    end
end
function potential_energy(sys::SimulatedBifurcation{T, :aSB}, x::Vector{T}) where T
    return sum(xi -> xi^4 / 4 + sys.a/2 * xi^2, x) - sys.c0 * sum(coupling * x[src(e)] * x[dst(e)] for (e, coupling) in zip(edges(sys.g), sys.J))
end
function potential_energy(sys::SimulatedBifurcation{T, :bSB}, x::Vector{T}) where T
    return sum(xi -> sys.a/2 * xi^2, x) - sys.c0 * sum(coupling * x[src(e)] * x[dst(e)] for (e, coupling) in zip(edges(sys.g), sys.J))
end
function potential_energy(sys::SimulatedBifurcation{T, :dSB}, x::Vector{T}) where T
    return sum(xi -> sys.a/2 * xi^2, x) - sys.c0 * sum(coupling * (x[src(e)] * sign(x[dst(e)]) + x[dst(e)] * sign(x[src(e)])) for (e, coupling) in zip(edges(sys.g), sys.J))
end
kinetic_energy(::SimulatedBifurcation{T}, p::Vector{T}) where T = sum(abs2, p) / 2
function force!(f::Vector{T}, sys::SimulatedBifurcation{T, :aSB}, x::Vector{T}) where T
    f .= x.^3 .+ sys.a .* x
    for (e, coupling) in zip(edges(sys.g), sys.J)
        f[src(e)] -= sys.c0 * coupling * x[dst(e)]
        f[dst(e)] -= sys.c0 * coupling * x[src(e)]
    end
    return f
end
function force!(f::Vector{T}, sys::SimulatedBifurcation{T, :bSB}, x::Vector{T}) where T
    f .= sys.a .* x
    for (e, coupling) in zip(edges(sys.g), sys.J)
        f[src(e)] -= sys.c0 * coupling * x[dst(e)]
        f[dst(e)] -= sys.c0 * coupling * x[src(e)]
    end
    return f
end
function force!(f::Vector{T}, sys::SimulatedBifurcation{T, :dSB}, x::Vector{T}) where T
    f .= sys.a .* x
    for (e, coupling) in zip(edges(sys.g), sys.J)
        f[src(e)] -= sys.c0 * coupling * sign(x[dst(e)])
        f[dst(e)] -= sys.c0 * coupling * sign(x[src(e)])
    end
    return f
end
force(sys::SimulatedBifurcation{T}, x::Vector{T}) where T = force!(Vector{T}(undef, nv(sys.g)), sys, x)

struct SimulatedBifurcationState{T}
    x::Vector{T}
    p::Vector{T}
end

function energy(sys::SimulatedBifurcation{T}, state::SimulatedBifurcationState{T}) where T
    return potential_energy(sys, state.x) + kinetic_energy(sys, state.p)
end

# simulate with Stormer-Verlet integrator
function simulate_bifurcation!(state::SimulatedBifurcationState{T}, sys::SimulatedBifurcation{T}; nsteps::Int, dt::T, clamp::Bool=false, checkpoint_steps::Int=typemax(Int)) where T
    checkpoints = Vector{SimulatedBifurcationState{T}}()
    
    # Initial half-step for momentum
    f = force(sys, state.x)
    state.p .+= 0.5 * dt * f
    
    for i in 1:nsteps
        # Position update
        state.x .+= dt * state.p
        clip && clamp!(state.x, -1, 1)
        
        # Force calculation
        force!(f, sys, state.x)
        
        # Momentum update (full step, except at the end)
        if i < nsteps
            state.p .+= dt * f
        else
            # Final half-step for momentum
            state.p .+= 0.5 * dt * f
        end
        
        # Save checkpoint if needed
        if mod(i, checkpoint_steps) == 0
            push!(checkpoints, SimulatedBifurcationState{T}(copy(state.x), copy(state.p)))
        end
    end
    
    return state, checkpoints
end