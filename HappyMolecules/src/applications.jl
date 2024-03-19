module Applications

using Random, StaticArrays
using DocStringExtensions
using ..HappyMolecules

export lennard_jones_triple_point

# setup docstring format
DocStringExtensions.@template (FUNCTIONS, METHODS, MACROS) =
    """
    $(SIGNATURES)
    $(DOCSTRING)
    $(METHODLIST)
    """

"""
Case study in Chapter 4 of the book "Understanding Molecular Simulation, From Algorithms to Applications".
It is about the molecule dynamics simulation of a Lennard-Jones Fluid in a 3D periodic box.
The parameters are set close to the triple point.

### Keyword arguments
* `natoms` is the number of atoms.
* `temperature` is the initial temperature.
* `density` is the density of atoms.
* `Nt` is the number of tims steps.
* `Δt` is the time step.
* `seed` is the random seed.
* `gr_lastn` is the number of last n samples for collecting radial distribution.
"""
function lennard_jones_triple_point(;
        natoms::Int = 108,  # number of atoms
        temperature::Real = 0.728,   # initial temperature
        density::Real = 0.8442,   # density of particles
        Nt = 2000,
        Δt = 0.001,
        seed::Int = 2,
        gr_lastn::Int = 500,
    )
    Random.seed!(seed)

    # the box
    volume = natoms / density
    L = volume ^ (1/3)
    box = PeriodicBox(SVector(L, L, L))

    # initial status
    lattice_pos = uniform_locations(box, natoms)
    velocities = [rand(SVector{3, Float64}) .- 0.5 for _ = 1:natoms]
    rc = L/2

    # create a `MDRuntime` instance
    md = molecule_dynamics(; lattice_pos, velocities, box, temperature, rc, Δt, potential=LennardJones(; rc))

    # Q: how to match the initial potential energy?
    # Anderson thermalstat.
    # Nose-Hoover thermalstat, difficult but better.
    ps = Float64[]
    ks = Float64[]
    temps = Float64[]

    bin = Bin(0.0, L/2, 200)
    for j=1:Nt
        step!(md)
        push!(ps, mean_potential_energy(md))
        push!(ks, mean_kinetic_energy(md))
        push!(temps, HappyMolecules.temperature(md))
        if j > Nt - gr_lastn
            HappyMolecules.collect_gr!(md, bin)
        end
    end
    return (;
        runtime = md,
        potential_energy = ps,
        kinetic_energy = ks,
        radial_ticks = ticks(bin),
        radial_distribution = HappyMolecules.finalize_gr(md, bin, gr_lastn)
    )
end

end
