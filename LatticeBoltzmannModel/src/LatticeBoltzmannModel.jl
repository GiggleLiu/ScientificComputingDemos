module LatticeBoltzmannModel

# import packages
using LinearAlgebra

export Point, Point2D, Point3D
export D2Q9, LatticeBoltzmann, step!, equilibrium_density, momentum, curl, example_d2q9, density

include("point.jl")
include("fluid.jl")

end
