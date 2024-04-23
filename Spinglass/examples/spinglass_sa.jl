using Spinglass

@info "Simulated annealing for the spinglass model"
filename = pkgdir(Spinglass, "data", "example.txt")
sap = load_spinglass(filename)

@info "Loaded spinglass from: $filename, number of spins = $(size(sap.coupling, 1))"
tempscales = 10 .- (1:64 .- 1) .* 0.15
nupdate_each_temperature = 4000
nrun = 30
@info """Start annealing:
- Temperatures: $(tempscales)
- number of updates each temperature: $nupdate_each_temperature
- number of runs
"""
opt_cost, opt_config = anneal(nrun, sap, collect(tempscales), nupdate_each_temperature)

@info "- Optimal cost: $opt_cost (known optimal: 3858)
- Optimal configuration: $opt_config"