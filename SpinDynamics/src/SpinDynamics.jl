module SpinDynamics

using CairoMakie
using LinearAlgebra, StaticArrays, Graphs

export simulate!, ClassicalSpinSystem, random_spins, TrotterSuzuki
export visualize_spins_animation, visualize_spins

include("spindynamics.jl")
include("visualize.jl")

end