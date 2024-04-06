using GenericTensorNetworks, GenericTensorNetworks.Graphs, GenericTensorNetworks.OMEinsum
using Random; Random.seed!(42)

# The 3-regular graph
n = parse(Int, get(ENV, "NV", "230"))
graph = Graphs.random_regular_graph(n, 3)
@info "We use a 3-regular graph as a demo, which has $(nv(graph)) vertices and $(ne(graph)) edges."

# Visualize the 3-regular graph
locations = zip(GenericTensorNetworks.LuxorGraphPlot.spring_layout(graph; C=10)...)
show_graph(graph; locs=locations, format=:png, filename="regular-$n.png")
@info "The demo graph is saved as `regular-$n.png`, optimizing the contraction order..."

# An anti-ferromagnetic spin glass problem
jsonfile = "regular-$n.json"
J = fill(-1, ne(graph))
if !isfile(jsonfile)
    problem = SpinGlass(graph; J, optimizer=TreeSA())
else
    @info "loading optimized contraction order from $jsonfile"
    code = readjson(jsonfile)
    h = ZeroWeight()
    target = MaxCut(code, graph, [2*J[i] for i=1:Graphs.ne(graph)], [-2*h[i] for i=1:Graphs.nv(graph)], Dict{Int,Int}())
    problem = SpinGlass(target, J, h)
end
@info "We consider an anti-ferromagnetic spin glass problem on the 3-regular graph. Its couling constants are J = $(problem.J)."
# The output is a tensor network with optimized contraction order.
@info "The contraction complexity is $(contraction_complexity(problem))"

# The lowest energy of the spin glass problem.
Emin = solve(problem, SizeMin())[]
@info "The ground state energy is: $Emin"

# The spin glass problem is reduced to the weighted `MaxCut` problem for solving.
reduced_problem = problem.target
@info "The `SpinGlass` problem is reduced to the `MaxCut` problem with edge_weights = $(reduced_problem.edge_weights)."

# verification
reduced_result = solve(reduced_problem, SizeMin())[]
extract_result(problem, reduced_result)

# In the following, we discuss the `MaxCut` problem and its tensor network representation.
# the `MaxCut` problem contains a `code` field, which specifies the tensor network contraction order.
@info "The contraction order is $(reduced_problem.code)"

# the tensor labels are:
labels = OMEinsum.getixsv(reduced_problem.code)

# the tensor network is:
tropical_tensors = GenericTensorNetworks.generate_tensors(Tropical(-1.0), reduced_problem)
@info """The corresponding tropical tensors are: $(join(["$l => $t" for (l, t) in zip(labels, tropical_tensors)], "\n"))."""

comp = contraction_complexity(reduced_problem.code, uniformsize(reduced_problem.code, 2))
@info "The contraction complexity is:\n$comp"

# save the tensor network as a JSON file
writejson(jsonfile, reduced_problem.code)
@info "The tensor network topology is saved as `$jsonfile`"

# by inputting the tropical tensors into the tensor network, we can obtain the result.
reduced_result = inv(reduced_problem.code(tropical_tensors...)[])
# Note: tropical `inv` is equivalent to flipping the sign.
@info "The contraction result is $reduced_result, which is then converted to spin-glass solution $(extract_result(problem, reduced_result))."

# the ground state configuration
ground_state = solve(problem, SingleConfigMin(; bounded=true))[].c.data
Emin_verify = spinglass_energy(graph, ground_state)
@info "The ground state is $ground_state and the energy is $Emin_verify"

show_graph(graph; locs=locations, vertex_colors=[
        iszero(ground_state[i]) ? "white" : "red" for i=1:nv(graph)], format=:png, filename="regular-$(n)_ground_state.png")
@info "The ground state is saved as `regular-$(n)_ground_state.png`"
