using CairoMakie
using DelimitedFiles
using LaTeXStrings

# an example for testing
# temperature = 2.0, magnetic field = 0.0, lattice size = 100
model = IsingModel(100, 0.0, 1/2)

# Constructs the initial random spin configuration
spin = rand([-1,1], model.l, model.l)

@time result = simulate!(model, spin; nsteps_heatbath = 1000, nsteps_eachbin = 1000, nbins = 100)
filename = joinpath(@__DIR__, "res.dat")
write(filename, result)

# Load the data
data = readdlm(filename)
fig = Figure()
ax = Axis(fig[1, 1], xlabel="time")
legends = [L"energy/spin", L"(energy/spin)^2", L"|m|", L"m^2", L"m^4"]

for i=1:5
    lines!(ax, data[:, i], label=legends[i])
end
axislegend(ax)
fig
save("mdata.png", fig, px_per_unit=2)