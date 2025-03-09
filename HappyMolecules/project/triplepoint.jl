using HappyMolecules
using HappyMolecules.Applications: lennard_jones_triple_point
using Makie, CairoMakie

res = lennard_jones_triple_point()

# plot the radial distribution
lines(res.radial_ticks, res.radial_distribution)

# plot the potential energy and kinetic energy as a function of time.
times = LinRange(0, res.runtime.t, length(res.kinetic_energy))
plt.plot(times, res.potential_energy; label="Potential energy")
plt.plot(times, res.kinetic_energy; label="Kinetic energy")
plt.plot(times, res.potential_energy + res.kinetic_energy; label="Total energy", color="k", ls="--")
plt.xlabel("step")
plt.ylabel("Energy/N")
plt.legend()
plt.show()

# TODO:display the time evolution process
let
    filename = tempname() * ".mp4"
    fig = Figure(; resolution=(800, 800))
    ax = Axis3(fig[1,1]; aspect=:data)
    limits = CairoMakie.FRect3D((0, 0, 0),(box.dimensions...,))
    limits!(ax, limits)
	points = Observable([Point3f(x...,) for x in md.x])
	directions = Observable([Point3f(x/100...,) for x in md.field])
	scatter!(ax, points)
	arrows!(ax, points, directions; linewidth=0.02, arrowsize=0.1)
	record(fig, filename, 1:500; framerate = 30, sleep=true) do i
		for j=1:10
			step!(md)
		end
		points[] = [Point3f(mod.(x, box.dimensions)...,) for x in md.x]
		directions[] = [Point3f(x/100...,) for x in md.field]
	end
end