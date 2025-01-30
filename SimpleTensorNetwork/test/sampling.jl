using Test, SimpleTensorNetwork
using SimpleTensorNetwork: MPS, random_mps, right_canonicalize!, left_canonicalize!, get_psi, train, show_imgs, load_mnist, bond_dims, gen_samples, generate_images, plot_distribution
using LinearAlgebra

@testset "random mps" begin
    mps = random_mps(Float32, 784, 30)
    @test length(mps.tensors) == 784
    @test count(size(i) == (30, 2, 30) for i in mps.tensors) == 782
    @test bond_dims(mps) == [fill(30, 783)..., 1]
end

@testset "canonicalize" begin
    nsite = 20
    mps = random_mps(Float32, nsite, 30)
    mps2 = right_canonicalize!(deepcopy(mps))
    mps3 = left_canonicalize!(deepcopy(mps))
    data = rand(0:1, 11, nsite)
    psi1 = normalize!(vec(get_psi(mps, data)))
    psi2 = normalize!(vec(get_psi(mps2, data)))
    @test psi1 ≈ psi2
    psi3 = normalize!(vec(get_psi(mps3, data)))
    @test psi1 ≈ psi3
end

@testset "load data" begin
    m = 20   # number of images
    data = load_mnist()[1:m, :]  # Load and slice first m rows
    @test size(data) == (m, 784)
    fig = show_imgs(data, 2, 10)
    @test fig isa SimpleTensorNetwork.CairoMakie.Figure
end

@testset "sampling" begin
    mps = random_mps(Float32, 42, 30)
    samples = gen_samples(mps, 25)
    @test size(samples) == (25, 42)
end

data = load_mnist()[1:25, :]
mps = random_mps(Float32, size(data, 2), 30)
results = train(mps, data)
generate_images(results[1])
plot_distribution(results[1], data)
generate_images(results[2])
generate_images(results[3])
generate_images(results[4])
generate_images(results[5])
plot_distribution(results[5], data)
