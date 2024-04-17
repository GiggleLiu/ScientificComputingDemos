using GenericTensorNetworks, GenericTensorNetworks.Graphs, GenericTensorNetworks.OMEinsum
using GenericTensorNetworks.LuxorGraphPlot
using Random; Random.seed!(42)

# The 3-regular graph
n = parse(Int, get(ENV, "NV", "100"))
graph = Graphs.random_regular_graph(n, 3)
@info "We use a 3-regular graph as a demo, which has $(nv(graph)) vertices and $(ne(graph)) edges."

# Visualize the 3-regular graph
locs = render_locs(graph, Layout(:spring; optimal_distance=100))
show_graph(graph, locs; format=:png, filename="regular-$n.png")
@info "The demo graph is saved as `regular-$n.png`, optimizing the contraction order..."

# An anti-ferromagnetic spin glass problem
jsonfile = "regular-$n.json"
J = fill(1, ne(graph))
if !isfile(jsonfile)
    problem = GenericTensorNetwork(SpinGlass(graph, J); optimizer=TreeSA())
    # save the tensor network as a JSON file
    writejson(jsonfile, problem.code)
    @info "The tensor network topology is saved as `$jsonfile`"
else
    @info "loading optimized contraction order from $jsonfile"
    code = readjson(jsonfile)
    h = ZeroWeight()
    problem = GenericTensorNetwork(SpinGlass(graph, J, h), code, Dict{Int, Int}())
end

@info "We consider an anti-ferromagnetic spin glass problem on the 3-regular graph. Its couling constants are J = $(problem.problem.weights[1:ne(graph)])."
# The output is a tensor network with optimized contraction order.
@info "The contraction order is $(problem.code)"
@info "The contraction complexity is $(contraction_complexity(problem))"

show_einsum(problem.code; optimal_distance=100, format=:png, filename="regular-$(n)_einsum.png")
@info "The tensor network diagram is saved as `regular-$(n)_einsum.png`"

# The lowest energy of the spin glass problem.
Emin = solve(problem, SizeMin())[]
@info "The ground state energy is: $Emin"

# the tensor labels are:
labels = OMEinsum.getixsv(problem.code)

# the ground state configuration
ground_state = solve(problem, SingleConfigMin(; bounded=true))[].c.data
Emin_verify = spinglass_energy(graph, ground_state; J)
@info "The ground state is $ground_state and the energy is $Emin_verify"

show_graph(graph, locs;
        vertex_colors=[iszero(ground_state[i]) ? "white" : "red" for i=1:nv(graph)],
        format=:png, filename="regular-$(n)_ground_state.png")
@info "The ground state is saved as `regular-$(n)_ground_state.png`"
