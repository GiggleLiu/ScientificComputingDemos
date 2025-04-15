using LinearAlgebra
using CairoMakie
using Random
using Spinglass: Transverse, iterate_T

function CIM_E_tot(x1_grid, x2_grid, alpha)
    J = [0 1; 1 0]  # Note: actually using [0 -1; -1 0] according to comment
    E = zeros(size(x1_grid))
    for ii in 1:size(x1_grid, 1)
        for jj in 1:size(x1_grid, 2)
            E[ii,jj] = 0.25*(x1_grid[ii,jj]^4 + x2_grid[ii,jj]^4) - 
                       0.5*alpha*(x1_grid[ii,jj]^2 + x2_grid[ii,jj]^2) + 
                       0.5*x1_grid[ii,jj]*x2_grid[ii,jj]
            # E[ii,jj] = -0.5*alpha*(x1_grid[ii,jj]^2 + x2_grid[ii,jj]^2) - 0.2*x1_grid[ii,jj]*x2_grid[ii,jj]
        end
    end
    return E
end

function main()
    J = [0 1; 1 0]

    n_step = 200
    trials = 1
    beta = exp.(range(log(0.1), log(20), length=n_step))
    SEED = rand(1:10000)

    # println("seed=$SEED")

    # Assuming the transverse function is implemented in Julia
    tran_g = Transverse(J, beta, trials; gama=0, g=1, a_set=2, Delta_t=0.3, 
                        c0=0.2, dtype=Float64, seed=SEED, track_energy=true)
    energy, Track_g = iterate_T(tran_g)
    # Track_g is always a straight line

    SEED = 5884
    tran_r = Transverse(J, beta, trials; gama=1, g=0, a_set=2, Delta_t=0.3, 
                       c0=0.2, dtype=Float64, seed=SEED, track_energy=true)
    energy, Track_r = iterate_T(tran_r)

    SEED = 5884
    tran_OMG = Transverse(J, beta, trials; gama=1, g=0.07, a_set=2, Delta_t=0.3, 
                         c0=0.2, dtype=Float64, seed=SEED, track_energy=true)
    energy, Track_OMG = iterate_T(tran_OMG)

    alpha = -1
    # Define x1 and x2 ranges
    x1_min, x1_max = -1, 1
    x2_min, x2_max = -1, 1

    # Generate grid data
    x1 = range(x1_min, x1_max, length=200)
    x2 = range(x2_min, x2_max, length=200)
    x1_grid = [i for i in x1, j in x2]
    x2_grid = [j for i in x1, j in x2]

    # Calculate energy at each grid point
    energy = CIM_E_tot(x1_grid, x2_grid, alpha)

    # Create figure with subplots
    fig = Figure(size=(1200, 700))

    # First row of plots (alpha = -1)
    ax1 = Axis(fig[1, 1], xlabel=L"x_1", ylabel=L"x_2", 
              title="(a)", titlealign=:left)
    hm1 = heatmap!(ax1, x1, x2, energy, colormap=:viridis)
    lines!(ax1, Track_g[1:101, 1], Track_g[1:101, 2], color=:white)
    scatter!(ax1, [Track_g[1, 1]], [Track_g[1, 2]], color=:white, marker='+', markersize=15)
    scatter!(ax1, [Track_g[151, 1]], [Track_g[151, 2]], color=:red, marker='+', markersize=15)
    xlims!(ax1, -1.02, 1.02)
    ylims!(ax1, -1.02, 1.02)
    Colorbar(fig[1, 1, Right()], hm1, label=L"E_{tot}")

    ax2 = Axis(fig[1, 2], xlabel=L"x_1", 
              title="(c)", titlealign=:left)
    hm2 = heatmap!(ax2, x1, x2, energy, colormap=:viridis)
    lines!(ax2, Track_r[1:101, 1], Track_r[1:101, 2], color=:white)
    scatter!(ax2, [Track_r[1, 1]], [Track_r[1, 2]], color=:white, marker='+', markersize=15)
    scatter!(ax2, [Track_r[101, 1]], [Track_r[101, 2]], color=:red, marker='+', markersize=15)
    xlims!(ax2, -1.02, 1.02)
    ylims!(ax2, -1.02, 1.02)
    Colorbar(fig[1, 2, Right()], hm2, label=L"E_{tot}")

    ax3 = Axis(fig[1, 3], xlabel=L"x_1", 
              title="(e)", titlealign=:left)
    hm3 = heatmap!(ax3, x1, x2, energy, colormap=:viridis)
    lines!(ax3, Track_OMG[1:101, 1], Track_OMG[1:101, 2], color=:white)
    scatter!(ax3, [Track_OMG[1, 1]], [Track_OMG[1, 2]], color=:white, marker='+', markersize=15)
    scatter!(ax3, [Track_OMG[101, 1]], [Track_OMG[101, 2]], color=:red, marker='+', markersize=15)
    xlims!(ax3, -1.02, 1.02)
    ylims!(ax3, -1.02, 1.02)
    Colorbar(fig[1, 3, Right()], hm3, label=L"E_{tot}")

    # Second row of plots (alpha = 1)
    energy = CIM_E_tot(x1_grid, x2_grid, 1)

    ax4 = Axis(fig[2, 1], xlabel=L"x_1", ylabel=L"x_2", 
              title="(b)", titlealign=:left)
    hm4 = heatmap!(ax4, x1, x2, energy, colormap=:viridis)
    lines!(ax4, Track_g[101:201, 1], Track_g[101:201, 2], color=:white)
    scatter!(ax4, [Track_g[101, 1]], [Track_g[101, 2]], color=:white, marker='+', markersize=15)
    scatter!(ax4, [Track_g[201, 1]], [Track_g[201, 2]], color=:red, marker='+', markersize=15)
    xlims!(ax4, -1.02, 1.02)
    ylims!(ax4, -1.02, 1.02)
    Colorbar(fig[2, 1, Right()], hm4, label=L"E_{tot}")

    ax5 = Axis(fig[2, 2], xlabel=L"x_1", 
              title="(d)", titlealign=:left)
    hm5 = heatmap!(ax5, x1, x2, energy, colormap=:viridis)
    lines!(ax5, Track_r[101:201, 1], Track_r[101:201, 2], color=:white)
    scatter!(ax5, [Track_r[101, 1]], [Track_r[101, 2]], color=:white, marker='+', markersize=15)
    scatter!(ax5, [Track_r[201, 1]], [Track_r[201, 2]], color=:red, marker='+', markersize=15)
    xlims!(ax5, -1.02, 1.02)
    ylims!(ax5, -1.02, 1.02)
    Colorbar(fig[2, 2, Right()], hm5, label=L"E_{tot}")

    ax6 = Axis(fig[2, 3], xlabel=L"x_1", 
              title="(f)", titlealign=:left)
    hm6 = heatmap!(ax6, x1, x2, energy, colormap=:viridis)
    lines!(ax6, Track_OMG[101:201, 1], Track_OMG[101:201, 2], color=:white)
    scatter!(ax6, [Track_OMG[101, 1]], [Track_OMG[101, 2]], color=:white, marker='+', markersize=15)
    scatter!(ax6, [Track_OMG[201, 1]], [Track_OMG[201, 2]], color=:red, marker='+', markersize=15)
    xlims!(ax6, -1.02, 1.02)
    ylims!(ax6, -1.02, 1.02)
    Colorbar(fig[2, 3, Right()], hm6, label=L"E_{tot}")

    save(joinpath(@__DIR__, "2_spin_plot.png"), fig)
    display(fig)
end

main()
