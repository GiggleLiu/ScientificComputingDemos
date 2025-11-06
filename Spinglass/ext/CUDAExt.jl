module CUDAExt

using CUDA, Spinglass
using Spinglass: SpinGlassSA
using CUDA.GPUArrays: @kernel, get_backend, @index

"""
    CUDA.cu(sa::SpinGlassSA) -> SpinGlassSA{T, CuMatrix{T}}

Transfer a spin glass problem to GPU memory.

# Arguments
- `sa::SpinGlassSA`: Problem on CPU

# Returns
- GPU version of the problem with coupling matrix in GPU memory

See also: `cpu`
"""
CUDA.cu(sa::SpinGlassSA) = SpinGlassSA(CUDA.CuArray(sa.coupling))

"""
    cpu(sa::SpinGlassSA) -> SpinGlassSA{T, Matrix{T}}

Transfer a spin glass problem back to CPU memory.

# Arguments
- `sa::SpinGlassSA`: Problem (possibly on GPU)

# Returns
- CPU version of the problem with coupling matrix in regular memory

See also: `CUDA.cu`
"""
cpu(sa::SpinGlassSA) = SpinGlassSA(Matrix(sa.coupling))

"""
    BatchedSpinConfig{T1, T2, MT1<:AbstractMatrix{T1}, MT2<:AbstractMatrix{T2}}

Batched spin configuration for parallel GPU annealing.

# Fields
- `config::MT1`: Matrix of spin configurations, size (N, nrun)
- `field::MT2`: Matrix of effective fields, size (N, nrun)

# Description
Stores multiple spin configurations as columns of matrices, enabling parallel
processing on GPU. Each column represents one independent annealing run.

See also: `anneal`, `SpinGlassSA`
"""
struct BatchedSpinConfig{T1, T2, MT1<:AbstractMatrix{T1}, MT2<:AbstractMatrix{T2}}
    config::MT1
    field::MT2
end

"""
    Spinglass.anneal(nrun::Int, prob::SpinGlassSA{TF, <:CuMatrix{TF}}, 
                     tempscales::CuVector{TF}, num_update_each_temp::Int) where {TF}

GPU-accelerated simulated annealing with parallel runs.

# Arguments
- `nrun::Int`: Number of parallel annealing runs
- `prob::SpinGlassSA{TF, <:CuMatrix{TF}}`: Problem on GPU
- `tempscales::CuVector{TF}`: Temperature schedule on GPU
- `num_update_each_temp::Int`: MC updates per temperature

# Returns
- `opt_cost::Float64`: Best energy found
- `opt_config::SpinConfig`: Configuration achieving the best energy

# Description
Runs `nrun` independent annealing processes in parallel on GPU. Each CUDA thread
handles one annealing run, with all runs synchronized at temperature changes.

# Algorithm
1. Generate random initial configurations on CPU
2. Transfer to GPU as batched config
3. Run parallel annealing kernel
4. Transfer results back to CPU
5. Find and return the best configuration

# Performance Tips
- Use `nrun` ≥ 128 to saturate GPU (multiples of 32 for best efficiency)
- Larger problems benefit more from GPU acceleration
- Ensure `prob` and `tempscales` are already on GPU

# Example
```julia
using CUDA, Spinglass

prob = load_spinglass("large_problem.txt")
prob_gpu = CUDA.cu(SpinGlassSA(prob))
temps_gpu = CUDA.cu(collect(10.0:-0.1:0.1))

# 256 parallel runs on GPU
opt_cost, opt_config = anneal(256, prob_gpu, temps_gpu, 5000)
```

See also: `anneal_run!`, `BatchedSpinConfig`
"""
function Spinglass.anneal(nrun::Int, prob::SpinGlassSA{TF, <:CuMatrix{TF}}, tempscales::CuVector{TF}, num_update_each_temp::Int) where {TF}
    # Generate initial configs on CPU
    initial_config = [random_config(cpu(prob)) for _ in 1:nrun]
    
    # Convert to batched format and transfer to GPU
    batch_config = BatchedSpinConfig(
        CUDA.CuArray(hcat(getfield.(initial_config, :config)...)), 
        CUDA.CuArray(hcat(getfield.(initial_config, :field)...))
    )
    
    # Run parallel annealing on GPU
    anneal_run!(batch_config, prob, tempscales, num_update_each_temp)
    
    # Transfer results back to CPU
    cpu_config = BatchedSpinConfig(Matrix(batch_config.config), Matrix(batch_config.field))
    
    # Find best configuration
    eng, idx = findmin(i -> Spinglass.energy(SpinConfig(cpu_config.config[:, i], cpu_config.field[:, i]), cpu(prob)), 1:nrun)
    return eng, SpinConfig(cpu_config.config[:, idx], cpu_config.field[:, idx])
end

"""
    anneal_run!(config, prob, tempscales, num_update_each_temp)

Execute GPU kernel for parallel batched simulated annealing.

# Arguments
- `config::BatchedSpinConfig`: Batched configurations on GPU (modified in-place)
- `prob::SpinGlassSA`: Problem with coupling matrix on GPU
- `tempscales::CuVector`: Temperature schedule on GPU
- `num_update_each_temp::Int`: Monte Carlo updates per temperature

# Description
Launches a GPU kernel where each thread runs one independent annealing trajectory.
The kernel iterates through the temperature schedule, performing Metropolis updates
at each temperature.

# Parallelization Strategy
- **Parallel over runs**: Each GPU thread = one annealing run
- **Sequential within run**: MC updates and temperatures are sequential per thread
- **Independent RNG**: Each thread has independent random number generation

This strategy maximizes GPU utilization for the embarrassingly parallel
multi-run annealing problem.

# Kernel Launch Configuration
- Grid size: nrun threads
- Block size: Automatically determined by CUDA.jl
- No shared memory or synchronization between threads

See also: `Spinglass.anneal`, `propose`, `flip!`
"""
function anneal_run!(config::BatchedSpinConfig{TI, TF, <:CuMatrix{TI}, <:CuMatrix{TF}}, prob::SpinGlassSA{TF, <:CuMatrix{TF}}, tempscales::CuVector{TF}, num_update_each_temp::Int) where {TI, TF}
    # Define the GPU kernel
    @kernel function kernel(config, field, coupling)
        ibatch = @index(Global, Linear)  # Thread index = run index
        
        # Each thread performs full annealing for one run
        for temp in tempscales
            beta = inv(temp)
            for _ = 1:num_update_each_temp
                proposal, ΔE = propose(config, field, coupling, ibatch)
                # Metropolis acceptance with thread-local RNG
                if exp(-beta*ΔE) > CUDA.Random.rand()
                    flip!(config, field, proposal, coupling, ibatch)
                end
            end
        end
    end
    
    # Launch kernel with one thread per annealing run
    kernel(get_backend(config.config))(config.config, config.field, prob.coupling; 
                                       ndrange=size(config.config, 2))
end
 
"""
    propose(config, field, coupling, ibatch::Int) -> (Int, Float64)

GPU version of spin flip proposal for batched annealing.

# Arguments
- `config`: Configuration matrix on GPU
- `field`: Field matrix on GPU
- `coupling`: Coupling matrix on GPU
- `ibatch::Int`: Index of the annealing run (column index)

# Returns
- `ispin::Int`: Randomly selected spin to flip
- `ΔE::Float64`: Energy change if flip is accepted

# Performance
- Uses GPU random number generator (thread-safe)
- O(1) operation per proposal
- Marked `@inline` for performance

See also: `flip!`, `anneal_run!`
"""
@inline function propose(config, field, coupling, ibatch::Int)
    ispin = CUDA.Random.rand(1:size(coupling, 1))
    # Energy change: factor 4 = 2 (spin flip) × 2 (symmetric counting)
    ΔE = -field[ispin, ibatch] * config[ispin, ibatch] * 4
    ispin, ΔE
end

"""
    flip!(config, field, ispin::Int, coupling, ibatch::Int)

GPU version of spin flip and field update for batched annealing.

# Arguments
- `config`: Configuration matrix on GPU (modified in-place)
- `field`: Field matrix on GPU (modified in-place)
- `ispin::Int`: Index of spin to flip
- `coupling`: Coupling matrix on GPU
- `ibatch::Int`: Index of the annealing run (column index)

# Description
Flips one spin in one configuration and updates the corresponding fields.
Only modifies column `ibatch` of the config and field matrices.

# Performance
- In-place modification for efficiency
- O(N) field update (unavoidable for dense coupling)
- Uses `@inbounds` for speed (safe in GPU context)
- Marked `@inline` for performance

See also: `propose`, `anneal_run!`
"""
@inline function flip!(config, field, ispin::Int, coupling, ibatch::Int)
    # Flip the spin
    @inbounds config[ispin, ibatch] = -config[ispin, ibatch]
    
    # Update all fields for this configuration
    for i=1:size(coupling, 1)
        @inbounds field[i, ibatch] += 2 * config[ispin, ibatch] * coupling[i,ispin]
    end
end

@info "`CUDAExt` (for `Spinglass`) is loaded successfully."

end
