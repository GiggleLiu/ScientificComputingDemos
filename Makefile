JL = julia

init:
	$(JL) -e "using Pkg; dir=\"$${dir}\"; @assert isdir(dir); Pkg.activate(dir); Pkg.instantiate(); Pkg.activate(joinpath(dir, \"examples\")); Pkg.develop(path = dir); Pkg.instantiate(); Pkg.precompile();"
	@echo "environment initialized at: $${dir} and $${dir}/examples"

update:
	$(JL) -e "using Pkg; dir=\"$${dir}\"; @assert isdir(dir); Pkg.activate(dir); Pkg.update(); Pkg.activate(joinpath(dir, \"examples\")); Pkg.update(); Pkg.precompile();"
	@echo "environment updated at: $${dir} and $${dir}/examples"

test:
	@echo "testing package at: $${dir}"
	$(JL) -e "using Pkg; dir=\"$${dir}\"; @assert isdir(dir); Pkg.activate(dir); Pkg.test();"

example:
	@echo "running example at: $${dir}/examples/main.jl"
	$(JL) -e "using Pkg; dir=joinpath(\"$${dir}\", \"examples\"); @assert isdir(dir); Pkg.activate(dir); include(joinpath(dir, \"main.jl\"));"

testall:
	for d in CompressedSensing HappyMolecules ImageProcessing KernelPCA \
	            LatticeBoltzmannModel LatticeGasCA MyFirstPackage \
	            PhysicsSimulation SimpleLinearAlgebra Spinglass; do \
		dir=$${d} make init; \
		dir=$${d} make test; \
	done
	@echo 'all done'
