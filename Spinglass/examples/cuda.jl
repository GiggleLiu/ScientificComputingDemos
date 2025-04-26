using Spinglass, Test
using CUDA

@info "#### Example: GPU-accelerated Simulated Annealing with CUDA ####"

# Load a spin glass problem from file
filename = pkgdir(Spinglass, "data", "example.txt")
sap = load_spinglass(filename)
@info "Loaded spinglass from: $filename, number of spins = $(Spinglass.num_variables(sap))"

# Define temperature schedule for annealing
# Starting at 10 and decreasing by 0.15 for each of the 64 steps
tempscales = 10 .- (1:64 .- 1) .* 0.15 |> collect

# Convert the spin glass problem to run on CUDA GPU
cusap = SpinGlassSA(sap) |> CUDA.cu
@info "Transferred problem to GPU for acceleration"

# Configure annealing parameters
nrun = 30                               # Number of independent runs
nupdate_each_temperature = 4000         # Updates per temperature

@info """Start GPU annealing:
- Temperatures: from $(tempscales[1]) to $(tempscales[end])
- Number of updates each temperature: $nupdate_each_temperature
- Number of runs: $nrun
"""

# Run simulated annealing with:
# - 30 independent runs
# - Temperature schedule on GPU
# - 4000 updates per temperature step
opt_cost, opt_config = anneal(nrun, cusap, CUDA.CuVector(tempscales), nupdate_each_temperature)

@info """Annealing results:
- Optimal cost: $opt_cost (known optimal: -3858)
- Optimal configuration found: $(opt_config)
"""

# Verify we found the known optimal solution
@test opt_cost == -3858
