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

function example_system(n::Int, T::Float64)
    topology = cylinder(n, n)
    Jmax, Hmax = 2.0, 2.0
    Jt = TimeDependent(ones(ne(topology)), (J, t) -> (J .= Jmax * sin(π/2 * t / T)))
    ht = TimeDependent(fill(SVector(-5.0, 0.0, 0.0), nv(topology)), (h, t) -> (h .= Ref(SVector(-Hmax * (cos(π/2 * t / T)), 0.0, 0.0))))
    sys = ClassicalSpinSystem(topology, Jt, ht)
    return sys
end

function simulate_afm_grid(n::Int, T::Float64; dt=0.01)
    sys = example_system(n, T)
    spins = random_spins(nv(sys.topology); xbias=5.0)
    _, history = simulate!(spins, sys; nsteps=ceil(Int, T/dt), dt=dt, checkpoint_steps=10, algorithm=TrotterSuzuki{2}(sys.topology))
    return history
end

function measure_zz(spins)
    n = length(spins)
    return sum(spins[i][3] * spins[j][3] for i in 1:n, j in 1:n if i != j)/n/(n-1)
end

using CairoMakie
function plot_zz(sys, history)
    zz = [abs(measure_zz(st.spins)) for st in history]
    eng = [energy(SpinDynamics.instantiate(sys, st.time), st.spins)/nv(sys.topology) for st in history]
    
    fig = Figure(size=(800, 600))
    ax = Axis(fig[1, 1], 
              xlabel="Time Steps", 
              ylabel="ZZ Correlation",
              title="Spin-Spin Correlation Over Time",
              )
    
    lines!(ax, 1:length(zz), zz, linewidth=2, color=:blue, label="ZZ")
    lines!(ax, 1:length(eng), eng, linewidth=2, color=:red, label="Energy")
    axislegend(ax, position=:rb)
    return fig
end

   
# Visualize the pulse shapes
function plot_pulses(T)
    sys = example_system(6, T)
    Jt = sys.coupling
    ht = sys.bias
    fig = Figure(size=(800, 400))
    ax = Axis(fig[1, 1], 
                xlabel="Time", 
                ylabel="Amplitude",
                title="Pulse Shapes")
    
    times = range(0, T, length=100)
    J_values = first.(SpinDynamics.value!.(Ref(Jt), times))
    H_values = first.(SpinDynamics.value!.(Ref(ht), times))
        
    lines!(ax, times, J_values, linewidth=2, color=:blue, label="J(t)")
    lines!(ax, times, -getindex.(H_values, 1), linewidth=2, color=:red, label="H_x(t)")
    lines!(ax, times, -getindex.(H_values, 2), linewidth=2, color=:green, label="H_y(t)")
    lines!(ax, times, -getindex.(H_values, 3), linewidth=2, color=:purple, label="H_z(t)")
    
    axislegend(ax, position=:rb)
    fig
end

plot_pulses(100.0)
sys = example_system(6, 80.0)
history = simulate_afm_grid(6, 80.0);
plot_zz(sys, history)