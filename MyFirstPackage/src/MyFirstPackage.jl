module MyFirstPackage

using LinearAlgebra

# export interfaces
export Lorenz, integrate_step
export Point, Point2D, Point3D
export RungeKutta, Euclidean

include("point.jl")
include("lorenz.jl")

end
