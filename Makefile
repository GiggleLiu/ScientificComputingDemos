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
