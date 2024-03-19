JL = julia

init-%:
	$(JL) -e 'using Pkg; dir=joinpath("lib", "$*"); @assert isdir(dir); Pkg.activate(dir); Pkg.instantiate(); Pkg.activate(joinpath(dir, "examples")); Pkg.develop(path = dir); Pkg.instantiate(); Pkg.precompile();'

update-%:
	$(JL) -e 'using Pkg; dir=joinpath("lib", "$*"); @assert isdir(dir); Pkg.activate(dir); Pkg.update(); Pkg.activate(joinpath(dir, "examples")); Pkg.update(); Pkg.precompile();'

test-%:
	$(JL) -e 'using Pkg; dir=joinpath("lib", "$*"); @assert isdir(dir); Pkg.activate(dir); Pkg.test();'

example-%:
	$(JL) -e 'using Pkg; dir=joinpath("lib", "$*", "examples"); @assert isdir(dir); Pkg.activate(dir); include(joinpath(dir, "example.jl"));'