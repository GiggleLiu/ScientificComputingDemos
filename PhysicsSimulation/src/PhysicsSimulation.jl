module PhysicsSimulation

using LinearAlgebra

export Point, Point2D, Point3D
export Body, NewtonSystem, LeapFrogSystem, step!, solar_system, leapfrog_simulation

include("point.jl")
include("planet.jl")

end
