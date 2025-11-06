"""
    sk_model(n::Int, seed::Int) -> Matrix{Float64}

Generate a Sherrington-Kirkpatrick (SK) spin glass model with Gaussian couplings.

# Arguments
- `n::Int`: Number of spins in the system
- `seed::Int`: Random seed for reproducibility

# Returns
- `J::Matrix{Float64}`: Symmetric n×n coupling matrix with zero diagonal

# Description
The SK model is a fully-connected spin glass where coupling constants J_{ij}
are drawn independently from a Gaussian distribution with:
- Mean: 0
- Variance: 1/n (scaled to make energy extensive)

The Hamiltonian is: H = -Σᵢ<ⱼ J_{ij}σᵢσⱼ

# Model Properties
- Mean-field model (all spins interact with all others)
- Exhibits a spin glass phase transition at T_c ≈ 1
- Classic example of frustration and disorder in statistical physics
- NP-hard to find ground states for large n

# Examples
```julia
# Generate a 100-spin SK model
J = sk_model(100, 42)

# Check properties
using LinearAlgebra
@assert issymmetric(J)
@assert all(diag(J) .== 0)
println("Mean coupling: ", mean(J[triu(trues(100,100), 1)]))
println("Std coupling: ", std(J[triu(trues(100,100), 1)]))
```

# References
- Sherrington & Kirkpatrick, Phys. Rev. Lett. 35, 1792 (1975)

See also: `Transverse`, `iterate_T`
"""
function sk_model(n::Int, seed::Int)
    variance = 1 / n
    Random.seed!(seed)
    
    # Generate upper triangular random couplings
    J = randn(n, n) .* sqrt(variance)
    J = triu(J, 1)  # Keep only upper triangle (no diagonal)
    
    # Make symmetric
    J = J + J'
    return J
end

"""
    Transverse{T<:AbstractFloat}

Quantum-inspired dynamics solver for spin glass optimization using transverse field oscillators.

# Description
Implements a coupled oscillator dynamics approach inspired by quantum annealing.
Each spin is represented as an oscillator variable x_i ∈ [-1, 1], evolving under
equations of motion that include:
- Classical spin glass coupling terms
- Transverse field (quantum tunneling analog)
- Damping and noise for thermalization
- Nonlinear terms to confine x_i to ±1

The dynamics gradually reduces the transverse field strength, allowing the system
to settle into low-energy classical spin configurations.

# Fields

## Model Parameters
- `J::Array{T}`: Coupling matrix (N×N)
- `beta::Array{T}`: Inverse temperature schedule
- `N::Int`: Number of spins
- `trials::Int`: Number of parallel trajectories
- `N_step::Int`: Number of time steps

## Physics Parameters
- `w2::T`: Sum of squared couplings (Σ J²)
- `m::T`: Mean coupling (Σ J / 2)
- `J_var::T`: Standard deviation of couplings
- `sum_w::T`: Energy offset
- `gama::T`: Damping coefficient
- `Delta_t::T`: Time step size
- `g::T`: Friction parameter
- `c0::T`: Coupling strength coefficient
- `a_set::T`: Final transverse field strength

## State Variables
- `x::Array{T}`: Position variables (spin proxies), size (trials × N)
- `y::Array{T}`: Momentum variables, size (trials × N)
- `z::Array{T}`: Auxiliary field cache (x·J)
- `cache::Array{T}`: Temporary storage for updates
- `a::Array{T}`: Transverse field schedule
- `a0::T`: Initial transverse field strength

## Configuration
- `seed::Int`: Random seed for initialization
- `track_energy::Bool`: Whether to track energy history

# Constructor
    Transverse(J, beta, trials; gama, g, a_set, Delta_t=0.1, c0=0.5, 
               dtype=Float64, seed=1, track_energy=true)

# Arguments
- `J`: Coupling matrix
- `beta`: Inverse temperature schedule (usually not used in dynamics)
- `trials`: Number of independent trajectories to run in parallel
- `gama`: Transverse field strength (quantum tunneling rate)
- `g`: Damping/friction coefficient
- `a_set`: Target value for transverse field parameter
- `Delta_t`: Integration time step (default: 0.1)
- `c0`: Coupling coefficient (default: 0.5)
- `dtype`: Floating point type (default: Float64)
- `seed`: Random seed (default: 1)
- `track_energy`: Track energy during evolution (default: true)

# Examples
```julia
# Generate SK model
J = sk_model(100, 42)

# Create transverse field model
beta = zeros(1000)  # Not actively used in dynamics
model = Transverse(J, beta, 10;  # 10 parallel trials
    gama=2.0,    # Transverse field strength
    g=1.0,       # Damping
    a_set=-0.5   # Final field parameter
)

# Run dynamics
energy, track = iterate_T(model)
```

# Physics Background
This method simulates the adiabatic evolution of a quantum transverse field Ising model:

    H(t) = -Σᵢⱼ Jᵢⱼσᵢᶻσⱼᶻ - Γ(t)Σᵢ σᵢˣ

where Γ(t) is the transverse field that starts large and decreases to zero.
The classical oscillator dynamics approximate this quantum evolution.

# References
- Santoro et al., Science 295, 2427 (2002) - Quantum annealing introduction
- Perdomo-Ortiz et al., Sci. Rep. 2, 571 (2012) - Continuous-time quantum walks

See also: `iterate_T`, `sk_model`, `calculate_energy`
"""
mutable struct Transverse{T<:AbstractFloat}
    # Model parameters
    J::Array{T}
    beta::Array{T}
    N::Int
    trials::Int
    N_step::Int
    
    # Physics parameters
    w2::T
    m::T
    J_var::T
    sum_w::T
    gama::T
    Delta_t::T
    g::T
    c0::T
    a_set::T
    
    # State variables
    x::Array{T}
    y::Array{T}
    z::Array{T}
    cache::Array{T}
    a::Array{T}
    a0::T
    
    # Configuration
    seed::Int
    track_energy::Bool
    
    function Transverse(
        J, beta, trials; gama, g, a_set,
        Delta_t=0.1, c0=0.5, dtype=Float64, seed=1, track_energy=true
    )
        # Convert J to specified dtype
        J_dev = convert.(dtype, J)
        w2 = sum(J_dev .* J_dev)
        
        # Convert beta to device
        beta_dev = convert.(dtype, beta)
        
        # Extract dimensions
        N = size(J, 1)
        m = sum(J) / 2
        N_step = length(beta)
        J_var = sqrt(sum(J .* J) / (N * (N - 1)))
        sum_w = -sum(J) / 4
        
        # Create empty arrays for state variables (initialized in iterate_T)
        empty_array_2d = zeros(dtype, 0, 0)
        empty_array_1d = zeros(dtype, 0)
        
        return new{dtype}(
            J_dev, beta_dev, N, trials, N_step,
            dtype(w2), dtype(m), dtype(J_var), dtype(sum_w), 
            dtype(gama), dtype(Delta_t), dtype(g), dtype(c0), dtype(a_set),
            empty_array_2d, empty_array_2d, empty_array_2d, empty_array_2d,
            empty_array_1d, dtype(1),
            seed, track_energy
        )
    end
end

"""
    calculate_energy(model::Transverse) -> Array

Calculate the classical spin glass energy for current oscillator positions.

# Arguments
- `model::Transverse`: The transverse field model with current state

# Returns
- Energy array of size (trials,) or (trials, 1)

# Description
Computes the classical Ising energy E = 0.5 Σᵢⱼ Jᵢⱼ xᵢ xⱼ where xᵢ are the
current oscillator positions (continuous analogs of spins). The factor of 0.5
accounts for double-counting in the sum.

# Note
This energy is meaningful when x values are close to ±1 (classical spin states).
During the transverse field dynamics, x values interpolate smoothly between
configurations, so the energy represents a "soft" classical energy.

See also: `Transverse`, `iterate_T`
"""
function calculate_energy(model::Transverse)
    return 0.5 .* sum((model.x[:, 1:model.N] * model.J) .* model.x[:, 1:model.N], dims=2)
end

"""
    initialize_state!(model::Transverse{T}) where {T}

Initialize the oscillator state variables for dynamics.

# Arguments
- `model::Transverse{T}`: The model to initialize (modified in-place)

# Description
Sets up initial conditions for the dynamics:
- Position (x) and momentum (y) initialized near zero with small random noise
- Transverse field parameter schedule `a` ramped from -2 to `a_set`
- Schedule is clamped to not exceed a₀ = 1

The small initial values allow the system to start in a "quantum superposition"
state before the transverse field drives it toward classical configurations.

# Side Effects
Modifies `model.x`, `model.y`, `model.a`, and `model.a0` in-place.

See also: `iterate_T`, `update_step!`
"""
function initialize_state!(model::Transverse{T}) where {T}
    Random.seed!(model.seed)
    
    # Initialize positions and momenta with small random noise
    model.x = randn(T, model.trials, model.N) .* T(0.01)
    model.y = randn(T, model.trials, model.N) .* T(0.01)
    
    # Set up transverse field annealing schedule
    model.a0 = T(1)
    model.a = range(T(-2), model.a_set, length=model.N_step) |> collect .|> T
    
    # Apply constraint: a should not exceed a0
    model.a[model.a .> model.a0] .= model.a0
end

"""
    update_step!(model::Transverse{T}, jj::Int) where {T}

Perform one time step of the coupled oscillator dynamics.

# Arguments
- `model::Transverse{T}`: The model state (modified in-place)
- `jj::Int`: Current time step index (used for transverse field schedule)

# Description
Updates the oscillator variables (x, y) according to coupled equations of motion:

    dx/dt = -g·F + γ·Δt·y
    dy/dt = -g·Δt·y - γ·F

where F = Δt·(x³ - a[jj]·x + c·z) and z = x·J is the interaction field.

The dynamics include:
- Nonlinear term (x³) to confine oscillators to ±1
- Transverse field term (-a[jj]·x) enabling tunneling
- Classical coupling term (c·z) from spin glass interactions
- Damping (g) and momentum coupling (γ)

After each update, x is clamped to [-1, 1] to maintain physical bounds.

# Side Effects
Modifies `model.x`, `model.y`, `model.z`, and `model.cache` in-place.

See also: `initialize_state!`, `iterate_T`
"""
function update_step!(model::Transverse{T}, jj::Int) where {T}
    # Calculate interaction field
    model.z = model.x * model.J
    c_val = model.track_energy ? model.c0 : T(0.5)
    
    # Compute force: nonlinear + transverse field + coupling
    model.cache = model.Delta_t .* (model.x .* model.x .* model.x .- 
                                     model.a[jj] .* model.x .+ 
                                     c_val .* model.z)
    
    # Update position and momentum using symplectic-like integrator
    model.x = model.x .- model.g .* model.cache .+ model.gama .* model.Delta_t .* model.y
    model.y = model.y .- model.g .* model.Delta_t .* model.y .- model.gama .* model.cache
    
    # Enforce bounds on oscillator positions
    model.x = min.(max.(model.x, -1), 1)
end

"""
    iterate_T(model::Transverse{T}) where {T} -> Tuple or Array

Run the transverse field dynamics simulation.

# Arguments
- `model::Transverse{T}`: The initialized model

# Returns
If `model.track_energy == true`:
- `energy`: Array of size (trials, N_step) with energy at each time step
- `Track`: Array of size (N_step+1, N) tracking one trajectory (if trials==1)

If `model.track_energy == false`:
- Final energy array of size (trials,)

# Description
Executes the full dynamical evolution of the coupled oscillator system.
The transverse field parameter `a[jj]` is gradually changed according to
the schedule, allowing the system to evolve from a "quantum" superposition
state to a classical low-energy configuration.

# Algorithm
1. Initialize oscillator states near origin
2. For each time step:
   - Update positions and momenta via equations of motion
   - Optionally record energy and trajectory
3. Return final or tracked energies

# Usage Patterns

**Optimization**: Run with many trials, keep best energy trajectory
```julia
model = Transverse(J, zeros(1000), 100; gama=2.0, g=1.0, a_set=-0.5)
energy, _ = iterate_T(model)
best_trial = argmin(energy[:, end])
best_energy = energy[best_trial, end]
```

**Analysis**: Single trajectory with tracking
```julia
model = Transverse(J, zeros(1000), 1; gama=2.0, g=1.0, a_set=-0.5)
energy, track = iterate_T(model)
# Plot energy vs time, visualize trajectory
```

See also: `Transverse`, `calculate_energy`, `update_step!`
"""
function iterate_T(model::Transverse{T}) where {T}
    initialize_state!(model)
    
    if model.track_energy
        # Initialize energy history and trajectory tracker
        energy = zeros(T, model.trials, model.N_step)
        Track = zeros(T, model.N_step+1, model.N)
        
        # Run dynamics with tracking
        for jj in 1:model.N_step
            update_step!(model, jj)
            
            # Record trajectory for single-trial runs
            if model.trials == 1
                Track[jj+1, :] = model.x
            end
            
            # Record energy at this time step
            energy[:, jj] = calculate_energy(model)
        end
        
        return energy, Track
    else
        # Run dynamics without tracking (faster)
        for jj in 1:model.N_step
            update_step!(model, jj)
        end
        
        return calculate_energy(model)
    end
end