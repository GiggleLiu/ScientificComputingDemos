using CairoMakie
using DelimitedFiles
using IsingModel

##### Single site update Monte Carlo method #####

function run_ising_simulation(lattice_size, temperature, magnetic_field)
    @info """Monte Carlo Simulation of Ferromagnetic Ising Model:
    - temperature = $temperature
    - magnetic_field = $magnetic_field
    - lattice_size = $lattice_size
    """
    model = IsingSpinModel(lattice_size, magnetic_field, 1/temperature)

    # Constructs the initial random spin configuration
    spin = rand([-1,1], model.l, model.l)

    nsteps_heatbath = 1000
    nsteps_eachbin = 1000
    nbins = 100
    @info """Start the simulation...
    - Monte Carlo steps: ($(nsteps_heatbath) steps heat bath) + ($(nsteps_eachbin) steps each bin) * ($nbins bins))
    """
    result = @time simulate!(model, spin; nsteps_heatbath, nsteps_eachbin, nbins)
    filename = joinpath(@__DIR__, "res.dat")
    write(filename, result)
    @info "Simulation is finished, and the data is saved to: `$filename`"
    
    return filename
end

function plot_simulation_data(filename, output_filename)
    @info "Plot the data..."
    data = readdlm(filename)
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="time")
    legends = ["energy/spin", "(energy/spin)²", "|m|", "m²", "m⁴"]

    for i=1:5
        lines!(ax, data[:, i], label=legends[i])
    end
    axislegend(ax)
    
    save(output_filename, fig, px_per_unit=2)
    @info "The plot is saved to: `$output_filename`."
    
    return fig
end

function generate_ising_video(model_type, lattice_size, temperature, magnetic_field, filename_prefix)
    @info "Generate the video for temperature = $temperature..."
    model = model_type(lattice_size, magnetic_field, 1/temperature)
    
    # animation
    fig = Figure()
    spin = rand([-1,1], model.l, model.l)
    
    if model_type == SwendsenWangModel
        config = SwendsenWangConfig(spin)
        spinobs = Observable(spin)
        text_x_pos = -20
    else
        spinobs = Observable(spin)
        text_x_pos = -30
    end
    
    ax1 = Axis(fig[1, 1]; aspect = DataAspect()); hidedecorations!(ax1); hidespines!(ax1)
    Makie.heatmap!(ax1, spinobs)
    txt = Observable("t = 0")
    Makie.text!(ax1, text_x_pos, lattice_size-10; text=txt, color=:black, fontsize=30, strokecolor=:white)
    
    filename = joinpath(@__DIR__, "$(filename_prefix)-$temperature.mp4")
    record(fig, filename, 2:1000; framerate = 24) do i
        if model_type == SwendsenWangModel
            mcstep!(model, config)
            spinobs[] = config.spin
        else
            mcstep!(model, spin)
            spinobs[] = spin
        end
        txt[] = "t = $(i-1)"
    end
    @info "The video is recorded in: $filename"
end

# Run single site update Monte Carlo simulation
lattice_size = 100
temperature = 2.0
magnetic_field = 0.0

data_file = run_ising_simulation(lattice_size, temperature, magnetic_field)
plot_simulation_data(data_file, joinpath(@__DIR__, "ising-data.png"))

# Generate videos for different temperatures
for temp in [0.2, 1.0, 3.0]
    generate_ising_video(IsingSpinModel, lattice_size, temp, magnetic_field, "ising-spins")
end

##### Swendsen-Wang algorithm #####

function run_swendsen_wang_simulation(lattice_size, temperature, magnetic_field)
    @info """Cluster Monte Carlo Simulation of Ferromagnetic Ising Model - The Swendsen-Wang Algorithm:
    - temperature = $temperature
    - magnetic_field = $magnetic_field
    - lattice_size = $lattice_size
    """
    model = SwendsenWangModel(lattice_size, magnetic_field, 1/temperature)

    # Constructs the initial random spin configuration
    spin = rand([-1,1], model.l, model.l)

    nsteps_heatbath = 1000
    nsteps_eachbin = 1000
    nbins = 100
    @info """Start the simulation...
    - Monte Carlo steps: ($(nsteps_heatbath) steps heat bath) + ($(nsteps_eachbin) steps each bin) * ($nbins bins))
    """
    result = @time simulate!(model, spin; nsteps_heatbath, nsteps_eachbin, nbins)
    filename = joinpath(@__DIR__, "sw.dat")
    write(filename, result)
    @info "Simulation is finished, and the data is saved to: `$filename`"
    
    return filename
end

# Run Swendsen-Wang simulation
sw_data_file = run_swendsen_wang_simulation(lattice_size, temperature, magnetic_field)
plot_simulation_data(sw_data_file, joinpath(@__DIR__, "sw-data.png"))

# Generate videos for different temperatures using Swendsen-Wang
for temp in [0.2, 1.0, 3.0]
    generate_ising_video(SwendsenWangModel, lattice_size, temp, magnetic_field, "swising-spins")
end