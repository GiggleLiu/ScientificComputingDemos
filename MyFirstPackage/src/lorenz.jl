"""
    Point{D, T}

A point in D-dimensional space, with coordinates of type T.

# Examples
```jldoctest
julia> p1 = Point(1.0, 2.0)
Point{2, Float64}((1.0, 2.0))

julia> p2 = Point(3.0, 4.0)
Point{2, Float64}((3.0, 4.0))

julia> p1 + p2
Point{2, Float64}((4.0, 6.0))
```
"""
struct Point{D, T <: Real}
    data::NTuple{D, T}
end
const Point2D{T} = Point{2, T}
const Point3D{T} = Point{3, T}
Point(x::Real...) = Point((x...,))
LinearAlgebra.dot(x::Point, y::Point) = mapreduce(*, +, x.data .* y.data)
Base.:*(x::Real, y::Point) = Point(x .* y.data)
Base.:*(x::Point, y::Real) = Point(x.data .* y)
Base.:/(y::Point, x::Real) = Point(y.data ./ x)
Base.:+(x::Point, y::Point) = Point(x.data .+ y.data)
Base.:-(x::Point, y::Point) = Point(x.data .- y.data)
Base.isapprox(x::Point, y::Point; kwargs...) = all(isapprox.(x.data, y.data; kwargs...))
Base.getindex(p::Point, i::Int) = p.data[i]
Base.broadcastable(p::Point) = p.data
Base.iterate(p::Point, args...) = iterate(p.data, args...)
Base.zero(::Type{Point{D, T}}) where {D, T} = Point(ntuple(i->zero(T), D))
Base.zero(::Point{D, T}) where {D, T} = Point(ntuple(i->zero(T), D))
distance(p::Point, q::Point) = sqrt(sum((p - q) .^ 2))

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