# Reduction to circuit satisfiability
@info "#### Example 1: Reduction to Circuit Satisfiability ####"
include("logic_gates.jl")

# Method 1: Generic tensor network
@info "#### Example 2: Generic Tensor Network Method (Exact) ####"
include("tropical_tensor_network.jl")

# Method 2: simulated annealing
@info "#### Example 3: Simulated Annealing Method (approximate) ####"
include("simulated_annealing.jl")