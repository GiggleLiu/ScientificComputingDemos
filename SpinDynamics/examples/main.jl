using SpinDynamics, Graphs, SpinDynamics.StaticArrays

function simulate_afm_grid(n::Int)
    topology = grid((n, n))
    sys = ClassicalSpinSystem(topology, ones(ne(topology)), fill(SVector(1.0, 0.0, 0.0), nv(topology)))
    spins = random_spins(nv(topology); xbias=5.0)
    _, history = simulate!(spins, sys; nsteps=1000, dt=0.01, checkpoint_steps=10, algorithm=TrotterSuzuki{2}(topology))
    return history
end

history = simulate_afm_grid(5)
visualize_spins_animation(vec([(i, j, 0) for i in 1:5 for j in 1:5]), history; filename=joinpath(@__DIR__, "spin_animation.mp4"))

function twopoint_model(kind::Symbol)
    g = SimpleGraph(2)
    add_edge!(g, 1, 2)
    return SimulatedBifurcation{kind}(g, ones(ne(g)); c0=0.2)
end
using CairoMakie
function visualize_energy_landscape!(ax, kind::Symbol, a)
    sys = twopoint_model(kind)
    sys.a = a
    # Create a grid of x and y values
    n = 100
    xmax = kind == :aSB ? 1.5 : 1.0
    x_range = range(-xmax, xmax, length=n)
    y_range = range(-xmax, xmax, length=n)
    
    # Calculate potential energy at each point
    energy_grid = SpinDynamics.potential_energy.(Ref(sys), [[x, y] for x in x_range, y in y_range])
    # Plot the energy as a heatmap
    hm = heatmap!(ax, x_range, y_range, energy_grid, colormap=:viridis)
    return hm
end

function visualize_energy_landscape_all()
    # Create the visualization
    fig = Figure(size=(800, 600))
    
    for (i, kind) in enumerate((:aSB, :bSB, :dSB))
        as = kind == :aSB ? (1.0, 0.0, -1.0) : (1.0, 0.5, 0.0)
        for j in 1:3
            a = as[j]
            ax = Axis(fig[j, 2*i-1], 
              xlabel="x₁", ylabel="x₂",
              title="$kind (a = $a)")
            hm = visualize_energy_landscape!(ax, kind, a)
            Colorbar(fig[j, 2*i], hm)
        end
    end
    return fig
end

visualize_energy_landscape_all()

function twopoint_bifurcation_simulation(kind::Symbol, initial)
    sys = twopoint_model(kind)
    dt = 0.01
    t = kind == :aSB ? 200 : 100
    clamp = kind == :aSB ? false : true
    state = SimulatedBifurcationState(randn(nv(sys.g)) .* initial, randn(nv(sys.g)) .* initial)
    state, history = simulate_bifurcation!(state, sys; nsteps=round(Int, t/dt), dt=dt, checkpoint_steps=10, clamp=clamp)
    return state, history, sys
end

function twopoint_energy_evolution()
    # Create the figure
    fig = Figure(size=(800, 600))
    # Create a subplot for each bifurcation type
    for (i, kind) in enumerate((:aSB, :bSB, :dSB))
        ax = Axis(fig[i, 1], 
                  xlabel="Time", 
                  ylabel="Energy",
                  title="Energy Evolution in $kind")
     
        state, history, sys = twopoint_bifurcation_simulation(kind, 0.05)
        # Extract time points and energy values
        times = getfield.(history, :time)
        potential_energies = getfield.(history, :potential_energy)
        kinetic_energies = getfield.(history, :kinetic_energy)
        total_energies = potential_energies .+ kinetic_energies
        
        # Plot the energy components
        lines!(ax, times, potential_energies, label="Potential Energy", linewidth=2)
        lines!(ax, times, total_energies, label="Total Energy", linewidth=3, color=:black)
        
        # Add legend to each subplot
        axislegend(ax, position=:lb)

        # show landscape and trajectory
        a = kind == :aSB ? -1.0 : 0.0
        ax = Axis(fig[i, 2], 
                  xlabel="x₁", ylabel="x₂",
                  title="$kind (a = $a)")
        hm = visualize_energy_landscape!(ax, kind, a)
        xs = [s.state.x[1] for s in history]
        ys = [s.state.x[2] for s in history]
        lines!(ax, xs, ys, linewidth=2, color=:white)
    end
    return fig
end
# Plot the energy evolution for the bifurcation model
energy_fig = twopoint_energy_evolution()
