using SpinDynamics, Graphs

# m: number of unit cells in the non-periodic direction (x-direction)
# n: number of unit cells in the periodic direction (y-direction)
function cylinder(m::Int, n::Int)
    graph = SimpleGraph(m * n)
    lis = LinearIndices((m, n))
    for i in 1:m, j in 1:n
        add_edge!(graph, lis[i, j], lis[i, mod1(j + 1, n)])
        i != m && add_edge!(graph, lis[i, j], lis[i + 1, j])
    end
    return graph
end

function simulate_afm_grid(n::Int, t::Float64; dt=0.01)
    topology = cylinder(n, n)
    sys = ClassicalSpinSystem(topology, ones(ne(topology)))
    spins = random_spins(nv(topology); xbias=5.0)
    _, history = simulate!(spins, sys; nsteps=ceil(Int, t/dt), dt=dt, checkpoint_steps=10, algorithm=TrotterSuzuki{2}(topology))
    return history
end

function measure_zz(spins)
    n = length(spins)
    return sum(spins[i][3] * spins[j][3] for i in 1:n, j in 1:n if i != j)/n/(n-1)
end

using CairoMakie
function plot_zz(history)
    zz = [measure_zz(spins) for spins in history]
    
    fig = Figure(size=(800, 600))
    ax = Axis(fig[1, 1], 
              xlabel="Time Steps", 
              ylabel="ZZ Correlation",
              title="Spin-Spin Correlation Over Time")
    
    lines!(ax, 1:length(zz), zz, linewidth=2, color=:blue, label="ZZ")
    axislegend(ax, position=:rb)
    return fig
end

history = simulate_afm_grid(6, 7.0)
plot_zz(history)