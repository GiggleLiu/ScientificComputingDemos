function sk_model(n, seed)
    # n=100
    variance = 1 / n
    # seed = 20
    Random.seed!(seed)
    J = randn(n, n) .* sqrt(variance)
    
    J = triu(J, 1)
    J = J + J'
    return J
end

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
        
        N = size(J, 1)
        m = sum(J) / 2
        N_step = length(beta)
        J_var = sqrt(sum(J .* J) / (N * (N - 1)))
        sum_w = -sum(J) / 4
        
        # Create empty arrays for state variables (will be initialized in iterate_T)
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

# Energy calculation
function calculate_energy(model::Transverse)
    return 0.5 .* sum((model.x[:, 1:model.N] * model.J) .* model.x[:, 1:model.N], dims=2)
end

# Initialize state
function initialize_state!(model::Transverse{T}) where {T}
    Random.seed!(model.seed)
    
    # Initialize x and y
    model.x = randn(T, model.trials, model.N) .* T(0.01)
    model.y = randn(T, model.trials, model.N) .* T(0.01)
    
    model.a0 = T(1)
    model.a = range(T(-2), model.a_set, length=model.N_step) |> collect .|> T
    
    # Apply constraint a <= a0
    model.a[model.a .> model.a0] .= model.a0
end

# Update step
function update_step!(model::Transverse{T}, jj::Int) where {T}
    model.z = model.x * model.J
    c_val = model.track_energy ? model.c0 : T(0.5)
    model.cache = model.Delta_t .* (model.x .* model.x .* model.x .- model.a[jj] .* model.x .+ c_val .* model.z)
    
    model.x = model.x .- model.g .* model.cache .+ model.gama .* model.Delta_t .* model.y
    model.y = model.y .- model.g .* model.Delta_t .* model.y .- model.gama .* model.cache
    
    # Clamp x values between -1 and 1
    model.x = min.(max.(model.x, -1), 1)
end

# Iterate with optional energy tracking
function iterate_T(model::Transverse{T}) where {T}
    initialize_state!(model)
    
    if model.track_energy
        # Initialize energy and Track
        energy = zeros(T, model.trials, model.N_step)
        Track = zeros(T, model.N_step+1, model.N)
        
        for jj in 1:model.N_step
            update_step!(model, jj)
            
            if model.trials == 1
                Track[jj+1, :] = model.x
            end
            
            energy[:, jj] = calculate_energy(model)
        end
        
        return energy, Track
    else
        # Just run the simulation without tracking
        for jj in 1:model.N_step
            update_step!(model, jj)
        end
        
        return calculate_energy(model)
    end
end