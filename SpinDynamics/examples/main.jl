using SpinDynamics, Graphs, CairoMakie
using SpinDynamics: SVector

# Example 1: Antiferromagnetic grid simulation
# Reference: Tsai, S.-H., Landau, D.P., 2008. Spin Dynamics: An Atomistic Simulation Tool for Magnetic Systems. Computing in Science & Engineering 10, 72–79. https://doi.org/10.1109/MCSE.2008.12

function simulate_afm_grid(n::Int; nsteps=1000, dt=0.01, checkpoint_interval=10)
    @info "Starting antiferromagnetic grid simulation with $(n)x$(n) grid..."
    
    # Create a grid topology
    topology = grid((n, n))
    @info "Created grid topology with $(nv(topology)) vertices and $(ne(topology)) edges"
    
    # Set up the spin system with uniform coupling and x-direction bias
    sys = ClassicalSpinSystem(
        topology, 
        ones(ne(topology)), 
        fill(SVector(1.0, 0.0, 0.0), nv(topology))
    )
    @info "Initialized spin system with uniform coupling and x-direction bias"
    
    # Initialize spins with bias in x-direction
    spins = random_spins(nv(topology); xbias=5.0)
    @info "Initialized random spins with x-direction bias"
    
    # Run the simulation
    @info "Running simulation for $(nsteps) steps with dt=$(dt)..."
    _, history = simulate!(
        spins, 
        sys; 
        nsteps=nsteps, 
        dt=dt, 
        checkpoint_steps=checkpoint_interval, 
        algorithm=TrotterSuzuki{2}(topology)
    )
    @info "Simulation complete with $(length(history)) checkpoints"
    
    return history
end

function visualize_spins(locs::Vector, spins::Vector) where T
    @info "Visualizing $(length(spins)) spins..."
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

function visualize_spins_animation(locs::Vector, history::Vector; filename::String) where T
    @info "Creating spin animation with $(length(history)) frames..."
    fig = Figure(size=(800, 600))
    ax = Axis3(fig[1, 1], aspect=:data, 
               xlabel="x", ylabel="y", zlabel="z",
               title="Spin Visualization", protrusions = (0, 0, 0, 15), elevation=1, viewmode = :fit)
    
    # Set limits to ensure all spins are visible
    max_coord = maximum(maximum(abs.(loc)) for loc in locs) + 1.0
    limits!(ax, -max_coord, max_coord, -max_coord, max_coord, -max_coord, max_coord)
    hidedecorations!(ax)
    
    # Create observables for the animation
    current_spins = Observable(history[1].spins)
    
    # Create the arrows plot with observables
    arrows!(ax, 
            [loc[1] for loc in locs], [loc[2] for loc in locs], [loc[3] for loc in locs],
            @lift([spin[1]/10 for spin in $(current_spins)]), 
            @lift([spin[2]/10 for spin in $(current_spins)]), 
            @lift([spin[3]/10 for spin in $(current_spins)]),
            arrowsize=0.1, linewidth=0.05, color=:blue)
    
    # Create animation
    framerate = 30
    @info "Recording animation to $filename with framerate $framerate..."
    return record(fig, filename, 1:length(history); framerate=framerate) do frame_idx
        current_spins[] = history[frame_idx].spins
        setproperty!(ax, :title, "Spin Visualization - Frame $frame_idx")
    end
end

# Run the simulation and visualize
@info """=== Starting Example 1: Antiferromagnetic Grid Simulation ===
Reference: Tsai, S.-H., Landau, D.P., 2008. Spin Dynamics: An Atomistic Simulation Tool for Magnetic Systems. Computing in Science & Engineering 10, 72–79. https://doi.org/10.1109/MCSE.2008.12
"""
history = simulate_afm_grid(5)
visualize_spins_animation(
    vec([(i, j, 0) for i in 1:5 for j in 1:5]), 
    history; 
    filename=joinpath(@__DIR__, "spin_animation.mp4")
)
@info "Spin animation saved to $(joinpath(@__DIR__, "spin_animation.mp4"))"

# Example 2: Simulated Bifurcation
# Reference: Goto, H., Endo, K., Suzuki, M., Sakai, Y., Kanao, T., Hamakawa, Y., Hidaka, R., Yamasaki, M., Tatsumura, K., 2021. High-performance combinatorial optimization based on classical mechanics. Science Advances 7, eabe7953. https://doi.org/10.1126/sciadv.abe7953
@info """=== Starting Example 2: Simulated Bifurcation ===
Reference: Goto, H., Endo, K., Suzuki, M., Sakai, Y., Kanao, T., Hamakawa, Y., Hidaka, R., Yamasaki, M., Tatsumura, K., 2021. High-performance combinatorial optimization based on classical mechanics. Science Advances 7, eabe7953. https://doi.org/10.1126/sciadv.abe7953
"""

# Create a simple two-point model for different bifurcation types
function twopoint_model(kind::Symbol; coupling_strength=0.2)
    @info "Creating two-point model for $kind bifurcation with coupling strength $coupling_strength"
    g = SimpleGraph(2)
    add_edge!(g, 1, 2)
    return SimulatedBifurcation{kind}(g, ones(ne(g)); c0=coupling_strength)
end

# Visualize the energy landscape for a given bifurcation type and parameter a
function visualize_energy_landscape!(ax, kind::Symbol, a; resolution=100)
    @info "Visualizing energy landscape for $kind bifurcation with parameter a=$a"
    sys = twopoint_model(kind)
    sys.a = a
    
    # Create a grid of x and y values
    xmax = kind == :aSB ? 1.5 : 1.0
    x_range = range(-xmax, xmax, length=resolution)
    y_range = range(-xmax, xmax, length=resolution)
    
    # Calculate potential energy at each point
    @info "Calculating potential energy on a $(resolution)x$(resolution) grid..."
    energy_grid = SpinDynamics.potential_energy.(Ref(sys), [[x, y] for x in x_range, y in y_range])
    
    # Plot the energy as a heatmap
    hm = heatmap!(ax, x_range, y_range, energy_grid, colormap=:viridis)
    return hm
end

# Visualize energy landscapes for all bifurcation types with different parameters
function visualize_energy_landscape_all()
    @info "Creating energy landscape visualizations for all bifurcation types..."
    fig = Figure(size=(1000, 700))
    
    for (i, kind) in enumerate((:aSB, :bSB, :dSB))
        # Different parameter values for different bifurcation types
        as = kind == :aSB ? (1.0, 0.0, -1.0) : (1.0, 0.5, 0.0)
        
        for j in 1:3
            a = as[j]
            ax = Axis(fig[j, 2*i-1], 
                xlabel="x₁", ylabel="x₂",
                title="$kind (a = $a)")
            hm = visualize_energy_landscape!(ax, kind, a)
            Colorbar(fig[j, 2*i], hm, label="Potential Energy")
        end
    end
    
    return fig
end

# Generate and display the energy landscape visualization
@info "Generating energy landscape visualization..."
landscape_fig = visualize_energy_landscape_all()
save(joinpath(@__DIR__, "energy_landscapes.png"), landscape_fig)
@info "Energy landscape visualization saved to $(joinpath(@__DIR__, "energy_landscapes.png"))"

# Run a bifurcation simulation for a two-point model
function twopoint_bifurcation_simulation(kind::Symbol, initial_scale=0.05)
    @info "Running bifurcation simulation for $kind with initial scale $initial_scale"
    sys = twopoint_model(kind)
    
    # Set simulation parameters based on bifurcation type
    dt = 0.01
    simulation_time = kind == :aSB ? 200 : 100
    clamp_values = kind == :aSB ? false : true
    
    # Initialize with small random values
    state = SimulatedBifurcationState(
        randn(nv(sys.g)) .* initial_scale, 
        randn(nv(sys.g)) .* initial_scale
    )
    
    # Run the simulation
    @info "Simulating for $(round(Int, simulation_time/dt)) steps with dt=$dt..."
    state, history = simulate_bifurcation!(
        state, 
        sys; 
        nsteps=round(Int, simulation_time/dt), 
        dt=dt, 
        checkpoint_steps=10, 
        clamp=clamp_values
    )
    @info "Simulation complete with $(length(history)) checkpoints"
    
    return state, history, sys
end

# Visualize the energy evolution during bifurcation
function twopoint_energy_evolution()
    @info "Creating energy evolution visualization for all bifurcation types..."
    fig = Figure(size=(1000, 800))
    
    for (i, kind) in enumerate((:aSB, :bSB, :dSB))
        @info "Processing $kind bifurcation..."
        # Energy evolution plot
        ax_energy = Axis(fig[i, 1], 
            xlabel="Time", 
            ylabel="Energy",
            title="Energy Evolution in $kind Bifurcation")
     
        state, history, sys = twopoint_bifurcation_simulation(kind, 0.05)
        
        # Extract time points and energy values
        times = getfield.(history, :time)
        potential_energies = getfield.(history, :potential_energy)
        kinetic_energies = getfield.(history, :kinetic_energy)
        total_energies = potential_energies .+ kinetic_energies
        
        # Plot the energy components
        lines!(ax_energy, times, potential_energies, label="Potential Energy", linewidth=2)
        lines!(ax_energy, times, kinetic_energies, label="Kinetic Energy", linewidth=2, linestyle=:dash)
        lines!(ax_energy, times, total_energies, label="Total Energy", linewidth=3, color=:black)
        
        # Add legend
        axislegend(ax_energy, position=:lb)

        # Trajectory plot
        a = kind == :aSB ? -1.0 : 0.0
        ax_traj = Axis(fig[i, 2], 
            xlabel="x₁", ylabel="x₂",
            title="Trajectory in $kind (a = $a)")
        
        # Show energy landscape
        hm = visualize_energy_landscape!(ax_traj, kind, a)
        
        # Plot trajectory
        xs = [s.state.x[1] for s in history]
        ys = [s.state.x[2] for s in history]
        lines!(ax_traj, xs, ys, linewidth=2, color=:white)
        scatter!(ax_traj, [xs[1]], [ys[1]], color=:red, markersize=10, label="Start")
        scatter!(ax_traj, [xs[end]], [ys[end]], color=:green, markersize=10, label="End")
        
        axislegend(ax_traj, position=:lt)
    end
    
    return fig
end

# Generate and save the energy evolution visualization
@info "Generating energy evolution visualization..."
energy_fig = twopoint_energy_evolution()
save(joinpath(@__DIR__, "bifurcation_energy_evolution.png"), energy_fig)
@info "Energy evolution visualization saved to $(joinpath(@__DIR__, "bifurcation_energy_evolution.png"))"
@info "=== Examples completed successfully ==="
