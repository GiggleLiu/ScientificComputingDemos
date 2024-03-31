using Test, IsingModel, DelimitedFiles

@testset "lattice" begin
    function original_lattice(ll,nn,neighbor,bondspin,spinbond)
        for s0=1:nn
            x0=mod(s0-1,ll)
            y0=div(s0-1,ll)
            x1=mod(x0+1,ll)
            x2=mod(x0-1+ll,ll)
            y1=mod(y0+1,ll)
            y2=mod(y0-1+ll,ll)
            s1=1+x1+y0*ll
            s2=1+x0+y1*ll  
            s3=1+x2+y0*ll
            s4=1+x0+y2*ll
            neighbor[1,s0]=s1
            neighbor[2,s0]=s2
            neighbor[3,s0]=s3
            neighbor[4,s0]=s4
            bondspin[1,s0]=s0
            bondspin[2,s0]=s1
            bondspin[1,s0+nn]=s0
            bondspin[2,s0+nn]=s2
            spinbond[1,s0]=s0
            spinbond[2,s0]=s0+nn
            spinbond[3,s1]=s0
            spinbond[4,s2]=s0+nn
        end
        return nothing
    end
    neighbor, bondspin, spinbond = zeros(Int, 4, 9), zeros(Int, 2, 18), zeros(Int, 4, 9)
    original_lattice(3, 9, neighbor, bondspin, spinbond)
    neighbor_ = IsingModel.lattice(3)
    bondspin_, spinbond_ = IsingModel.spinbondmap(neighbor_)
    @test neighbor == neighbor_
    @test bondspin == bondspin_
    @test spinbond == spinbond_
end

@testset "fixed sized stack" begin
    stack = IsingModel.FixedSizedStack{Int}(10)
    @test isempty(stack)
    for i = 1:10
        push!(stack, i)
    end
    @test !isempty(stack)
    @test length(stack) == 10
    @test_throws BoundsError push!(stack, 11)
    for i = 10:-1:1
        @test pop!(stack) == i
    end
    @test isempty(stack)
    @test_throws BoundsError pop!(stack)

    for i = 1:10
        push!(stack, i)
    end
    IsingModel.reset!(stack)
    @test isempty(stack)
end

@testset "energy" begin
    model = SwendsenWangModel(10, 0.0, 0.1)
    spin = fill(-1, model.l, model.l)
    @test energy(model, spin) ≈ -200
    model = SwendsenWangModel(10, 0.1, 0.0)
    spin = fill(-1, model.l, model.l)
    @test energy(model, spin) ≈ -190
end

@testset "simulate and save" begin
    model = SwendsenWangModel(10, 0.1, 0.5)
    spin = rand([-1,1], model.l, model.l)
    result = simulate!(model, spin; nsteps_heatbath = 100, nsteps_eachbin = 100, nbins = 100)
    filename = joinpath(@__DIR__, "res.dat")
    write(filename, result)
    data = readdlm(filename)
    @testset "data" begin
        @test size(data) == (100, 5)
        @test all(data[:,2:5] .>= 0)
        @test all(data[:,1] .<= 0)
    end
end