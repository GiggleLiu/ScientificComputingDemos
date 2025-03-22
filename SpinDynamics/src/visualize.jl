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

function visualize_spins_animation(locs::Vector, history::Vector{Checkpoint{T}}; filename::String) where T
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
    return record(fig, filename, 1:length(history); framerate=framerate) do frame_idx
        current_spins[] = history[frame_idx].spins
        setproperty!(ax, :title, "Spin Visualization - Frame $frame_idx")
    end
end

