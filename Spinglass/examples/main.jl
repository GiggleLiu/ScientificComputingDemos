#####################################################
# Example 1: Reduction to Circuit Satisfiability
#####################################################
@info "#### Reduction to Circuit Satisfiability ####"
using Spinglass, Graphs, ProblemReductions, GenericTensorNetworks

# Demonstrate logic gates as spin glass models
# NOT gate
gadget_not = ProblemReductions.spinglass_gadget(Val(:¬))
@info "Logic NOT gate is represented by the spinglass model:\n$gadget_not"
gs_not = ProblemReductions.findbest(gadget_not.problem, BruteForce())
@info "Ground state energy and ground states of the NOT gate:\n$gs_not"

# AND gate
gadget_and = ProblemReductions.spinglass_gadget(Val(:∧))
@info "Logic AND gate is represented by the spinglass model:\n$gadget_and"
gs_and = ProblemReductions.findbest(gadget_and.problem, BruteForce())
@info "Ground state energy and ground states of the AND gate:\n$gs_and"

# OR gate
gadget_or = ProblemReductions.spinglass_gadget(Val(:∨))
@info "Logic OR gate is represented by the spinglass model:\n$gadget_or"
gs_or = ProblemReductions.findbest(gadget_or.problem, BruteForce())
@info "Ground state energy and ground states of the OR gate:\n$gs_or"

# Compose a 2-bit x 2-bit multiplier
@info "Composing a 2-bit x 2-bit multiplier..."
fact = ProblemReductions.Factoring(2, 2, 4)
paths = ProblemReductions.reduction_paths(Factoring, SpinGlass)
mapres = ProblemReductions.reduceto(paths[1], fact)

solution = ProblemReductions.findbest(target_problem(mapres), GenericTensorNetworks.GTNSolver())
@assert length(solution) == 1  # the solution is unique in this case
extracted = ProblemReductions.extract_solution(mapres, solution[1])
@info "Multiplier of 2 bits x 2 bits is represented by the spinglass model:\n$mapres"
@info "The extracted solution is: $extracted, decoded as $(ProblemReductions.read_solution(fact, extracted)), the multiplication of which should be $(fact.input)"

#####################################################
# Example 2: Simulated Annealing Method (approximate)
#####################################################
@info "#### Simulated Annealing for Spinglass solving ####"
using Spinglass

# Load a spin glass problem from file
filename = pkgdir(Spinglass, "data", "example.txt")
sap = load_spinglass(filename)
@info "Loaded spinglass from: $filename, number of spins = $(sap.n)"

# Configure annealing parameters
tempscales = 10 .- (1:64 .- 1) .* 0.15  # Temperature schedule
nupdate_each_temperature = 4000         # Updates per temperature
nrun = 30                               # Number of independent runs

@info """Start annealing:
- Temperatures: $(tempscales)
- Number of updates each temperature: $nupdate_each_temperature
- Number of runs: $nrun
"""

# Run simulated annealing
opt_cost, opt_config = anneal(nrun, sap, collect(tempscales), nupdate_each_temperature)

@info """Annealing results:
- Optimal cost: $opt_cost (known optimal: 3858)
- Optimal configuration: $opt_config
"""