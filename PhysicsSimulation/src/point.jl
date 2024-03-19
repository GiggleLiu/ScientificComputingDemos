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