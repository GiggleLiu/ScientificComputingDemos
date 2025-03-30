module SpinDynamics

using CairoMakie
using LinearAlgebra, StaticArrays, Graphs

export simulate!, ClassicalSpinSystem, random_spins, TrotterSuzuki, TimeDependent, energy
export SimulatedBifurcation, SimulatedBifurcationState, simulate_bifurcation!
export visualize_spins_animation, visualize_spins

include("simulation.jl")
include("simulated_bifurcation.jl")
include("visualize.jl")

end