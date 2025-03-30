module SpinDynamics

using LinearAlgebra, StaticArrays, Graphs

export simulate!, ClassicalSpinSystem, random_spins, TrotterSuzuki, TimeDependent, energy
export SimulatedBifurcation, SimulatedBifurcationState, simulate_bifurcation!

include("simulation.jl")
include("simulated_bifurcation.jl")

end