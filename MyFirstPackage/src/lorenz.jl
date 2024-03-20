struct Lorenz
    σ::Float64
    ρ::Float64
    β::Float64
end

function field(p::Lorenz, u)
    x, y, z = u
    Point(p.σ*(y-x), x*(p.ρ-z)-y, x*y-p.β*z)
end

abstract type AbstractIntegrator end
struct RungeKutta{K} <: AbstractIntegrator end
struct Euclidean <: AbstractIntegrator end

# Runge-Kutta 4th order method
function integrate_step(f, ::RungeKutta{4}, t, y, Δt)
    k1 = Δt * f(t, y)
    k2 = Δt * f(t+Δt/2, y + k1 / 2)
    k3 = Δt * f(t+Δt/2, y + k2 / 2)
    k4 = Δt * f(t+Δt, y + k3)
    return y + k1/6 + k2/3 + k3/3 + k4/6
end

# Euclidean integration
function integrate_step(f, ::Euclidean, t, y, Δt)
    return y + Δt * f(t, y)
end

function integrate_step(lz::Lorenz, int::AbstractIntegrator, u, Δt)
    return integrate_step((t, u) -> field(lz, u), int, zero(Δt), u, Δt)
end