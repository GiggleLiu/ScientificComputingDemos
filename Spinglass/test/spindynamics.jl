using Test
using Spinglass: ClassicalSpinSystem, SpinVector, simulate!, greedy_coloring, is_valid_coloring, partite_edges, single_spin_dynamics_operator, single_spin_dynamics, SVector, TrotterSuzuki, random_spins
using Graphs

@testset "Spin dynamics" begin
    topology = grid((3, 3))
    sys = ClassicalSpinSystem(topology, [1.0, 1.0, 1.0])
    spins = [SVector(ntuple(i -> randn(), 3)) for _ in 1:nv(sys.topology)]
    state, history = simulate!(spins, sys; nsteps=100, dt=0.1, checkpoint_steps=10, algorithm=TrotterSuzuki{2}(topology))
    @test length(history) == 10
    @test length(state) == 9
end

@testset "Greedy coloring" begin
    g = grid((10, 10))
    coloring = greedy_coloring(g)
    @test is_valid_coloring(g, coloring)
    @test length(unique(coloring)) <= 5
    eparts = partite_edges(g)
    @test length(eparts) <= 4

    g = path_graph(10)
    eparts = partite_edges(g)
    @test length(eparts) <= 2
end

@testset "single spin dynamics" begin
    #s = SVector(randn(), randn(), randn())
    s = SVector(1.0, 0.0, 0.0)
    field = SVector(randn(), randn(), randn())
    op = single_spin_dynamics_operator(field)
    @test op * s ≈ single_spin_dynamics(field, s)
end


using CairoMakie
function visualize_spins(locs::Vector, spins::Vector{SVector{3, T}}) where T
    fig = Figure(size=(800, 600))
    ax = Axis3(fig[1, 1], aspect=:data, 
               xlabel="x", ylabel="y", zlabel="z",
               title="Spin Visualization", protrusions = (0, 0, 0, 15), elevation=1, viewmode = :fit)
    
    arrows!(ax, 
            [loc[1] for loc in locs], [loc[2] for loc in locs], [loc[3] for loc in locs],
            [spin[1]/10 for spin in spins], [spin[2]/10 for spin in spins], [spin[3]/10 for spin in spins],
            arrowsize=0.1, linewidth=0.05, 
            color=:blue)
    
    # Set limits to ensure all spins are visible
    max_coord = maximum(maximum(abs.(loc)) for loc in locs) + 1.0
    limits!(ax, -max_coord, max_coord, -max_coord, max_coord, -max_coord, max_coord)
    hidedecorations!(ax)
    
    return fig
end

function visualize_spins_animation(locs::Vector, history::Vector{Vector{SVector{3, T}}}, filename="spin_animation.mp4") where T
    fig = Figure(size=(800, 600))
    ax = Axis3(fig[1, 1], aspect=:data, 
               xlabel="x", ylabel="y", zlabel="z",
               title="Spin Visualization", protrusions = (0, 0, 0, 15), elevation=1, viewmode = :fit)
    
    # Set limits to ensure all spins are visible
    max_coord = maximum(maximum(abs.(loc)) for loc in locs) + 1.0
    limits!(ax, -max_coord, max_coord, -max_coord, max_coord, -max_coord, max_coord)
    hidedecorations!(ax)
    
    # Create observables for the animation
    current_spins = Observable(history[1])
    frame_num = Observable(1)
    
    # Create the arrows plot with observables
    arrows!(ax, 
            [loc[1] for loc in locs], [loc[2] for loc in locs], [loc[3] for loc in locs],
            @lift([spin[1]/10 for spin in $(current_spins)]), 
            @lift([spin[2]/10 for spin in $(current_spins)]), 
            @lift([spin[3]/10 for spin in $(current_spins)]),
            arrowsize=0.1, linewidth=0.05, color=:blue)
    
    # Create animation
    framerate = 30
    return record(fig, filename, 1:length(history); framerate=framerate) do frame_idx
        current_spins[] = history[frame_idx]
        setproperty!(ax, :title, "Spin Visualization - Frame $frame_idx")
    end
end

topology = grid((3, 3))
sys = ClassicalSpinSystem(topology, [1.0, 1.0, 1.0])
spins = random_spins(nv(topology))
visualize_spins(vec([(i, j, 0) for i in 1:3 for j in 1:3]), spins)
# history = [random_spins(nv(topology)) for _ in 1:100]
_, history = simulate!(spins, sys; nsteps=100, dt=0.1, checkpoint_steps=1, algorithm=TrotterSuzuki{2}(topology))
visualize_spins_animation(vec([(i, j, 0) for i in 1:3 for j in 1:3]), history)