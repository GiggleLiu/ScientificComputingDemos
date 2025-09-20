JL = julia

init:
	$(JL) -e "using Pkg; dir=\"$${case}\"; @assert isdir(dir); Pkg.activate(dir); Pkg.instantiate(); Pkg.activate(joinpath(dir, \"examples\")); Pkg.develop(path = dir); Pkg.instantiate(); Pkg.precompile();"
	@echo "environment initialized at: $${case} and $${case}/examples"

update:
	$(JL) -e "using Pkg; dir=\"$${case}\"; @assert isdir(dir); Pkg.activate(dir); Pkg.update(); Pkg.activate(joinpath(dir, \"examples\")); Pkg.update(); Pkg.precompile();"
	@echo "environment updated at: $${case} and $${case}/examples"

test:
	@echo "testing package at: $${case}"
	$(JL) -e "using Pkg; dir=\"$${case}\"; @assert isdir(dir); Pkg.activate(dir); Pkg.test();"

example:
	@echo "running example at: $${case}/examples/main.jl"
	$(JL) -e "using Pkg; dir=joinpath(\"$${case}\", \"examples\"); @assert isdir(dir); Pkg.activate(dir); include(joinpath(dir, \"main.jl\"));"

testall:
	for case in CompressedSensing HappyMolecules ImageProcessing KernelPCA \
	            LatticeBoltzmannModel LatticeGasCA MyFirstPackage \
	            PhysicsSimulation SimpleLinearAlgebra Spinglass; do \
		case=$${case} make init; \
		case=$${case} make test; \
	done
	@echo 'all done'
