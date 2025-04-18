using ADSeismic, CairoMakie


nx = ny = 201
nstep = 1000
src = (nx÷2, ny÷5)
param = AcousticPropagatorParams(nx=nx, ny=ny, 
        nstep=nstep, dt=0.1/nstep,  dx=200/(nx-1), dy=200/(nx-1),
        Rcoef = 1e-8)
rc = Ricker(param, 30.0, 200.0, 1e6)  # source

# prepare the example landscape
function three_layers(nx, ny)
    layers = ones(nx+2, ny+2)
    n_piece = div(nx + 1, 3) + 1
    for k = 1:3
        i_interval = (k-1)*n_piece+1:min(k*n_piece, nx+2)
        layers[:, i_interval] .= 0.5 + (k-1)*0.25
    end
    return (3300 .* layers) .^ 2
end

c2 = three_layers(nx, ny)

detector_locs = CartesianIndex.([(50, 50), (50, 100), (50, 150),
         (150, 50), (150, 100), (150, 150)])

function show_landscape(landscape, detector_locs)
    # Visualize the velocity model with detector locations
    fig = Figure(size=(800, 600))
    ax = Axis(fig[1, 1], 
        xlabel="X (grid points)", 
        ylabel="Y (grid points)", 
        title="Landscape Model with Detector Locations")

    # Create heatmap of velocity model
    hm = heatmap!(ax, landscape, colormap=:viridis)
    Colorbar(fig[1, 2], hm, label="Velocity (m²/s²)")

    # Plot detector locations
    detector_x = [loc[1] for loc in detector_locs]
    detector_y = [loc[2] for loc in detector_locs]
    scatter!(ax, detector_y, detector_x, color=:red, markersize=15, marker=:star5)

    # Plot source location
    scatter!(ax, [src[2]], [src[1]], color=:silver, markersize=15, marker=:circle)
    text!(ax, src[2], src[1] - 10, text="Source", color=:white, align=(:center, :center))

    # Add legend
    elements = [
        MarkerElement(color=:red, marker=:star5, markersize=15),
        MarkerElement(color=:silver, marker=:circle, markersize=15)
    ]
    labels = ["Detector", "Source"]
    Legend(fig[2, 1:2], elements, labels, orientation=:horizontal)
    save(joinpath(@__DIR__, "velocity_model.png"), fig)
    fig
end

show_landscape(c2, detector_locs)


target_pulses = solve_detector(param, src, rc, c2, detector_locs)

# Plot the target pulses
function plot_detector_signals(pulses, detector_locs)
    fig = Figure(size=(800, 600))
    ax = Axis(fig[1, 1], 
        xlabel="Time step", 
        ylabel="Amplitude", 
        title="Detector Signals")
    
    # Plot each detector's signal as a separate line
    for i in 1:size(pulses, 1)
        lines!(ax, 1:size(pulses, 2), pulses[i, :], 
               label="Detector $(i) ($(detector_locs[i][1]), $(detector_locs[i][2]))")
    end
    
    # Add legend
    Legend(fig[1, 2], ax, "Detector Signals")
    
    # Save the figure
    save(joinpath(@__DIR__, "detector_signals.png"), fig)
    
    return fig
end

# Generate the plot
detector_plot = plot_detector_signals(target_pulses, detector_locs)


c20 = 3300^2*ones(nx+2,ny+2)

# loss is |u[:,40,:]-ut[:,40,:]|^2
function getgrad_mse(; c2, param, src, srcv, target_pulses, detector_locs, treeverse_δ=20)
    nx, ny = size(c2) .- 2
    logger = ReversibleSeismic.TreeverseLog()
    s0 = ReversibleSeismic.SeismicState(Float64, nx, ny)
    gn = ReversibleSeismic.SeismicState(Float64, nx, ny)
    gn.u[size(c2,1)÷2,size(c2,2)÷2+20] -= 1.0
    res, (g_tv_x, g_tv_srcv, g_tv_c) = treeverse_solve_detector(0.0, s0;
        target_pulses=target_pulses, detector_locs=detector_locs,
        param=param, c=c2, src=src,
        srcv=srcv, δ=treeverse_δ, logger=logger)
    println(logger)
    return res.data[1], (g_tv_x.data[2].u, g_tv_srcv, g_tv_c), logger
end

loss, (gin, gsrcv, gc), log = getgrad_mse(c2=c20, param=param, src=src,
            srcv=rc, target_pulses=target_pulses, detector_locs=detector_locs,
            treeverse_δ=20, usecuda=false)