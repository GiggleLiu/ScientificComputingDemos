using SpinDynamics, Test, CairoMakie

@testset "visualize" begin
    spins = random_spins(10)
    locations = [(i, 0, 0) for i in 1:10]
    fig = visualize_spins(locations, spins)
    @test fig isa CairoMakie.Figure
    visualize_spins_animation(spins, [spins]; filename=joinpath(@__DIR__, "spin_animation.mp4"))
    @test isfile(joinpath(@__DIR__, "spin_animation.mp4"))
end
