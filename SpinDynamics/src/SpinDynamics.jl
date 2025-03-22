module SpinDynamics

using CairoMakie
using LinearAlgebra, StaticArrays, Graphs

export simulate!, ClassicalSpinSystem, random_spins, TrotterSuzuki, TimeDependent, energy
export visualize_spins_animation, visualize_spins

include("simulation.jl")
include("visualize.jl")

end