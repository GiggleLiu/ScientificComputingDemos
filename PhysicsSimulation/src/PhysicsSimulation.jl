module PhysicsSimulation

using LinearAlgebra, Graphs

export Point, Point2D, Point3D
export nsite, connections, EigenSystem, eigensystem, EigenModes, eigenmodes
export Body, NewtonSystem, LeapFrogSystem, step!, solar_system, leapfrog_simulation
export spring_chain, SpringSystem, coordinate, velocity, waveat

include("point.jl")
include("planet.jl")
include("chain.jl")

end
