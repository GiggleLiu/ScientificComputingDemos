module HappyMolecules

using StaticArrays
using Statistics
using DocStringExtensions

export Bin, ticks, ncounts
export molecule_dynamics, step!
export PeriodicBox, Box, random_locations, uniform_locations, volume
export PotentialField, LennardJones, potential_energy, force
export positions, velocities, forces, num_particles,
        mean_kinetic_energy, temperature, mean_potential_energy,
        pressure

# setup docstring format
DocStringExtensions.@template (FUNCTIONS, METHODS, MACROS) =
    """
    $(SIGNATURES)
    $(DOCSTRING)
    $(METHODLIST)
    """

include("Core.jl")
include("enzyme.jl")
include("applications.jl")

end
