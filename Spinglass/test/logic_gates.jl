using Test, Spinglass
using Spinglass: StaticElementVector

@testset "gates" begin
    res = ground_states(sg_gadget_and().sg)
    @test sort(res.c.data) == map(x->StaticElementVector(2, x), [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]])
    tt = truth_table(sg_gadget_and())
    @test length(tt) == 4
    @test tt[[0, 0]] == tt[[0, 1]] == tt[[1, 0]] == [0]
    @test tt[[1, 1]] == [1]

    res = ground_states(sg_gadget_or().sg)
    @test sort(res.c.data) == map(x->StaticElementVector(2, x), [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])

    res = ground_states(sg_gadget_not().sg)
    @test sort(res.c.data) == map(x->StaticElementVector(2, x), [[0, 1], [1, 0]])
end

@testset "arraymul" begin
    arr = sg_gadget_arraymul()
    tt = truth_table(arr)
    @test length(tt) == 16
    @test tt[[0, 0, 0, 0]] == tt[[1, 0, 0, 0]] == tt[[0, 1, 0, 0]] == [0, 0]
    @test tt[[0, 0, 1, 0]] == tt[[1, 0, 1, 0]] == tt[[0, 1, 1, 0]] == tt[[1, 1, 0, 0]] ==
        tt[[0, 0, 0, 1]] == tt[[1, 0, 0, 1]] == tt[[0, 1, 0, 1]]  == [0, 1]
    @test tt[[1, 1, 1, 0]] == tt[[1, 1, 0, 1]] ==
        tt[[1, 0, 1, 1]] == tt[[0, 1, 1, 1]] == tt[[0, 0, 1, 1]] == [1, 0]
    @test tt[[1, 1, 1, 1]] == [1, 1]
end

@testset "arraymul compose" begin
    arr = Spinglass.compose_multiplier(2, 2)
    @test arr.sg.n == 20
    tt = truth_table(arr)
    @test length(tt) == 16
    Spinglass.set_input!(arr, [0, 1, 0, 1])  # 2 x 2 == 4
    @test truth_table(arr) == Dict([0, 1, 0, 1] => [0, 0, 1, 0])
end