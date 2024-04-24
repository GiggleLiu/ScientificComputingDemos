JL = julia

init-%:
	$(JL) -e 'using Pkg; dir="$*"; @assert isdir(dir); Pkg.activate(dir); Pkg.instantiate(); Pkg.activate(joinpath(dir, "examples")); Pkg.develop(path = dir); Pkg.instantiate(); Pkg.precompile();'
	echo 'environment initialized at: $* and $*/examples'

update-%:
	$(JL) -e 'using Pkg; dir="$*"; @assert isdir(dir); Pkg.activate(dir); Pkg.update(); Pkg.activate(joinpath(dir, "examples")); Pkg.update(); Pkg.precompile();'
	echo 'environment updated at: $* and $*/examples'

test-%:
	echo 'testing package at: $*'
	$(JL) -e 'using Pkg; dir="$*"; @assert isdir(dir); Pkg.activate(dir); Pkg.test();'

example-%:
	echo 'running example at: $*/examples/main.jl'
	$(JL) -e 'using Pkg; dir=joinpath("$*", "examples"); @assert isdir(dir); Pkg.activate(dir); include(joinpath(dir, "main.jl"));'

testall: init-CompressedSensing test-CompressedSensing init-HappyMolecules test-HappyMolecules init-ImageProcessing test-ImageProcessing init-KernelPCA test-KernelPCA init-LatticeBolzmannModel test-LatticeBolzmannModel init-LatticeGasCA test-LatticeGasCA init-MyFirstPackage test-MyFirstPackage init-PhysicsSimulation test-PhysicsSimulation init-SimpleLinearAlgebra test-SimpleLinearAlgebra init-Spinglass test-Spinglass
	echo 'all done'