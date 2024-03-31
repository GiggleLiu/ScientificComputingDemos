using CairoMakie
using DelimitedFiles
using LaTeXStrings
using IsingModel

# an example for testing
lattice_size = 100
temperature = 2.0
magnetic_field = 0.0
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

@info "Plot the data..."
data = readdlm(filename)
fig = Figure()
ax = Axis(fig[1, 1], xlabel="time")
legends = [L"energy/spin", L"(energy/spin)^2", L"|m|", L"m^2", L"m^4"]

for i=1:5
    lines!(ax, data[:, i], label=legends[i])
end
axislegend(ax)
fig

filename = joinpath(@__DIR__, "mdata.png")
save(filename, fig, px_per_unit=2)
@info "The plot is saved to: `$filename`."

for temperature in [0.2, 1.0, 3.0]
    @info "Generate the video for temperature = $temperature..."
    model = IsingSpinModel(lattice_size, magnetic_field, 1/temperature)
    # animation
    fig = Figure()
    ax1 = Axis(fig[1, 1]; aspect = DataAspect()); hidedecorations!(ax1); hidespines!(ax1)  # hides ticks, grid and lables, and frame
    spin = rand([-1,1], model.l, model.l)
    spinobs = Observable(spin)
    Makie.heatmap!(ax1, spinobs, camera=campixel!)
    filename = joinpath(@__DIR__, "ising-spins-$temperature.mp4")
    record(fig, filename, 2:1000; framerate = 24) do i
        mcstep!(model, spin)
        spinobs[] = spin
    end
    @info "The video is recorded in: $filename"
end