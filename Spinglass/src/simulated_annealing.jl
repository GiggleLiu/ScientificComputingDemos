"""
    load_spinglass(filename::String) -> SpinGlass

Load a spin glass problem from a text file.

# Arguments
- `filename::String`: Path to the file containing spin glass data

# File Format
The input file should contain three columns:
1. First spin index (0-based indexing)
2. Second spin index (0-based indexing)
3. Coupling strength J_{ij}

Each row represents an edge in the interaction graph with its coupling constant.

# Returns
- `SpinGlass`: A spin glass problem instance from ProblemReductions.jl

# Examples
```julia
# Load from file where each line is: vertex1 vertex2 coupling_strength
sap = load_spinglass("data/example.txt")
println("Number of spins: ", ProblemReductions.num_variables(sap))
```

# Notes
- The function automatically converts 0-based indices to 1-based Julia indexing
- External fields are initialized to zero
- Duplicate edges are not checked; behavior with duplicates is undefined

See also: `SpinGlass`, `anneal`
"""
function load_spinglass(filename::String)
    data = readdlm(filename)
    # Convert from 0-based to 1-based indexing
    is = Int.(view(data, :, 1)) .+ 1
    js = Int.(view(data, :, 2)) .+ 1
    num_spin = max(maximum(is), maximum(js))
    
    # Build the interaction graph
    g = SimpleGraph(num_spin)
    for (i, j) in zip(is, js)
        add_edge!(g, i, j)
    end
    
    # Create SpinGlass with coupling strengths and zero external fields
    SpinGlass(g, data[:,3], zeros(num_spin))
end

"""
    SpinGlassSA{T, MT<:AbstractMatrix{T}}

Spin glass model representation optimized for simulated annealing.

# Fields
- `coupling::MT`: Symmetric coupling matrix where coupling[i,j] = J_{ij}/2 for efficiency

# Description
This structure stores the spin glass problem as a dense coupling matrix, which enables
fast energy calculations and field updates during simulated annealing. The coupling
matrix is symmetric, and each element stores half the actual coupling strength to avoid
double-counting in energy calculations.

# Constructor
    SpinGlassSA(sg::SpinGlass{<:SimpleGraph})

Convert a `SpinGlass` problem (graph representation) to a matrix representation.

# Examples
```julia
# Convert from graph to matrix representation
sg = load_spinglass("problem.txt")
sap = SpinGlassSA(sg)
println("Problem size: ", num_spin(sap))
```

See also: `SpinGlass`, `SpinConfig`, `anneal`
"""
struct SpinGlassSA{T, MT<:AbstractMatrix{T}}
    coupling::MT
end

function SpinGlassSA(sg::SpinGlass{<:SimpleGraph})
    coupling = zeros(nv(sg.graph), nv(sg.graph))
    for (e, weight) in zip(edges(sg.graph), sg.J)
        # Store half the coupling for symmetric energy calculation
        coupling[src(e), dst(e)] = weight/2
        coupling[dst(e), src(e)] = weight/2
    end
    SpinGlassSA(coupling)
end

"""
    num_spin(prob::SpinGlassSA) -> Int

Return the number of spins in the spin glass problem.

# Examples
```julia
sap = SpinGlassSA(load_spinglass("problem.txt"))
n = num_spin(sap)
```
"""
num_spin(prob::SpinGlassSA) = size(prob.coupling, 1)

"""
    SpinConfig{Ts, Tf}

Configuration state for simulated annealing of spin glass problems.

# Fields
- `config::Vector{Ts}`: Spin configuration where each element is ±1
- `field::Vector{Tf}`: Effective field on each spin (h_i = Σ_j J_{ij}σ_j)

# Description
This structure maintains both the current spin configuration and the precomputed
effective field on each spin. The field is updated incrementally during spin flips,
which makes energy difference calculations O(1) instead of O(N).

# Notes
- The field stores the local field contribution: h_i = Σ_j J_{ij}σ_j
- When a spin flips, the field must be updated for all neighboring spins
- This caching strategy significantly improves performance for large systems

See also: `random_config`, `flip!`, `energy`
"""
struct SpinConfig{Ts, Tf}
    config::Vector{Ts}
    field::Vector{Tf}
end

"""
    random_config(prob::SpinGlassSA) -> SpinConfig

Generate a random initial spin configuration.

# Arguments
- `prob::SpinGlassSA`: The spin glass problem instance

# Returns
- `SpinConfig`: A random configuration with spins ±1 and precomputed fields

# Description
Creates a random initial state where each spin is independently assigned +1 or -1
with equal probability. The effective field on each spin is computed as h_i = Σ_j J_{ij}σ_j,
which allows for efficient energy difference calculations during annealing.

# Examples
```julia
prob = SpinGlassSA(load_spinglass("problem.txt"))
config = random_config(prob)
E = energy(config, prob)
```

See also: `SpinConfig`, `energy`, `anneal`
"""
function random_config(prob::SpinGlassSA)
    config = rand([-1,1], num_spin(prob))
    # Precompute the effective field for each spin
    SpinConfig(config, prob.coupling*config)
end

"""
    anneal_singlerun!(config, prob, tempscales::Vector{Float64}, num_update_each_temp::Int) -> (Float64, SpinConfig)

Perform a single run of simulated annealing using Metropolis-Hastings updates.

# Arguments
- `config`: Initial spin configuration (will be modified in-place)
- `prob`: Spin glass problem with `energy`, `flip!`, and `propose` interfaces
- `tempscales::Vector{Float64}`: Temperature schedule (should be monotonically decreasing)
- `num_update_each_temp::Int`: Number of spin flip attempts at each temperature

# Returns
- `opt_cost::Float64`: Best (minimum) energy found during the run
- `opt_config`: Configuration corresponding to the optimal energy

# Algorithm
The Metropolis-Hastings algorithm accepts or rejects proposed spin flips based on:
- Always accept if ΔE < 0 (energy decreases)
- Accept with probability exp(-β·ΔE) if ΔE > 0 (Boltzmann distribution)

where β = 1/T is the inverse temperature.

# Performance Notes
- Uses `@simd` for potential SIMD optimization of the inner loop
- Tracks the best configuration found, not just the final configuration
- Updates are performed in-place for memory efficiency

# Examples
```julia
prob = SpinGlassSA(load_spinglass("problem.txt"))
config = random_config(prob)
tempscales = collect(10.0:-0.1:0.1)
opt_cost, opt_config = anneal_singlerun!(config, prob, tempscales, 1000)
```

See also: `anneal`, `propose`, `flip!`, `energy`
"""
function anneal_singlerun!(config, prob, tempscales::Vector{Float64}, num_update_each_temp::Int)
    cost = energy(config, prob)
    
    opt_config = deepcopy(config)
    opt_cost = cost
    
    for beta = 1 ./ tempscales
        # @simd hint for potential Single Instruction Multiple Data optimization
        @simd for _ = 1:num_update_each_temp
            proposal, ΔE = propose(config, prob)
            # Metropolis acceptance criterion
            if exp(-beta*ΔE) > rand()  # Accept move
                flip!(config, proposal, prob)
                cost += ΔE
                # Track the best configuration seen
                if cost < opt_cost
                    opt_cost = cost
                    opt_config = deepcopy(config)
                end
            end
        end
    end
    opt_cost, opt_config
end
 
"""
    anneal(nrun::Int, prob, tempscales::Vector{Float64}, num_update_each_temp::Int) -> (Float64, SpinConfig)

Perform simulated annealing optimization with multiple independent runs.

# Arguments
- `nrun::Int`: Number of independent annealing runs to perform
- `prob`: Spin glass problem (either `SpinGlass` or `SpinGlassSA`)
- `tempscales::Vector{Float64}`: Temperature schedule (decreasing sequence)
- `num_update_each_temp::Int`: Number of Monte Carlo updates at each temperature

# Returns
- `opt_cost::Float64`: Best (minimum) energy found across all runs
- `opt_config::SpinConfig`: Configuration achieving the optimal energy

# Description
Runs simulated annealing multiple times from different random initial conditions
and returns the best solution found. Multiple runs help avoid getting trapped in
poor local minima, which is especially important for hard optimization problems.

# Typical Temperature Schedules
- Linear: `T₀:-ΔT:T_final` (e.g., `10.0:-0.15:0.1`)
- Geometric: `T₀ * α^k` for k=0,1,2,... (slower cooling)
- Logarithmic: `T₀/log(1+k)` (very slow cooling)

# Performance Tips
- More runs (nrun) improve solution quality but increase runtime linearly
- Slower cooling (more temperatures) generally gives better results
- More updates per temperature help equilibration but cost more time
- For GPU acceleration, see `CUDAExt` extension

# Examples
```julia
using Spinglass

# Load problem
prob = load_spinglass("data/example.txt")

# Configure annealing
tempscales = collect(10.0:-0.15:0.55)  # 64 temperature steps
nupdate = 4000                          # Updates per temperature
nrun = 30                               # Independent runs

# Solve
opt_cost, opt_config = anneal(nrun, prob, tempscales, nupdate)
println("Optimal energy: ", opt_cost)
```

See also: `anneal_singlerun!`, `load_spinglass`, `SpinGlassSA`
"""
anneal(nrun::Int, prob::SpinGlass, tempscales::Vector{Float64}, num_update_each_temp::Int) = 
    anneal(nrun, SpinGlassSA(prob), tempscales, num_update_each_temp)

function anneal(nrun::Int, prob::SpinGlassSA, tempscales::Vector{Float64}, num_update_each_temp::Int)
    local opt_config, opt_cost
    for r = 1:nrun
        initial_config = random_config(prob)
        cost, config = anneal_singlerun!(initial_config, prob, tempscales, num_update_each_temp)
        if r == 1 || cost < opt_cost
            opt_cost = cost
            opt_config = config
        end
    end
    opt_cost, opt_config
end

"""
    energy(config::SpinConfig, sap::SpinGlassSA) -> Float64

Calculate the total energy of a spin configuration.

# Arguments
- `config::SpinConfig`: Current spin configuration
- `sap::SpinGlassSA`: Spin glass problem instance

# Returns
- Energy value: E = Σᵢⱼ J_{ij}σᵢσⱼ

# Description
Computes the total energy using the quadratic form σᵀJσ where σ is the spin
configuration vector and J is the coupling matrix. Since J stores J_{ij}/2,
the formula automatically handles the factor of 1/2 to avoid double-counting.

# Complexity
O(N²) where N is the number of spins (dense matrix multiplication)

# Examples
```julia
prob = SpinGlassSA(load_spinglass("problem.txt"))
config = random_config(prob)
E = energy(config, prob)
```

See also: `propose`, `flip!`
"""
energy(config::SpinConfig, sap::SpinGlassSA) = sum(config.config'*sap.coupling*config.config)

"""
    propose(config::SpinConfig, ::SpinGlassSA) -> (Int, Float64)

Propose a random spin flip and compute the energy change.

# Arguments
- `config::SpinConfig`: Current spin configuration (not modified)
- Spin glass problem (argument not used, included for interface consistency)

# Returns
- `ispin::Int`: Index of the spin to flip
- `ΔE::Float64`: Energy change if the flip is accepted

# Description
Randomly selects a spin to flip and computes the energy change using the cached
field values. The energy difference for flipping spin i is:

    ΔE = -4 h_i σ_i

where the factor 4 comes from:
- Factor 2 from σ_i → -σ_i (sign flip)
- Factor 2 from counting interactions twice (i→j and j→i)

# Performance
- O(1) operation due to field caching (vs O(N) without caching)
- Marked `@inline` for compiler optimization
- Uses `@inbounds` to skip bounds checking (safe in this context)

See also: `flip!`, `energy`
"""
@inline function propose(config::SpinConfig, ::SpinGlassSA)
    ispin = rand(1:length(config.config))
    # Calculate energy change: factor of 4 = 2 (spin flip) × 2 (symmetric counting)
    @inbounds ΔE = -config.field[ispin] * config.config[ispin] * 4
    ispin, ΔE
end

"""
    flip!(config::SpinConfig, ispin::Int, sap::SpinGlassSA) -> SpinConfig

Apply a spin flip and update the cached field values.

# Arguments
- `config::SpinConfig`: Configuration to modify (in-place)
- `ispin::Int`: Index of the spin to flip
- `sap::SpinGlassSA`: Spin glass problem instance

# Returns
- Modified `config` (same object, updated in-place)

# Description
Flips spin `ispin` from +1 to -1 or vice versa, then updates all field values
to maintain consistency. The field update is:

    h_j → h_j + 2σ_i J_{ij}

where σ_i is the NEW value of the flipped spin.

# Performance
- In-place modification for memory efficiency
- O(N) operation to update all fields
- Uses `@simd` for vectorization and `@inbounds` for speed
- Marked `@inline` for compiler optimization

# Examples
```julia
# After accepting a proposal
proposal, ΔE = propose(config, prob)
if exp(-beta*ΔE) > rand()
    flip!(config, proposal, prob)
end
```

See also: `propose`, `energy`
"""
@inline function flip!(config::SpinConfig, ispin::Int, sap::SpinGlassSA)
    # Flip the spin
    @inbounds config.config[ispin] = -config.config[ispin]
    
    # Update the effective field on all spins (incremental update)
    @simd for i=1:num_spin(sap)
        @inbounds config.field[i] += 2 * config.config[ispin] * sap.coupling[i,ispin]
    end
    config
end
