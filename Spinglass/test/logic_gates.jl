using Test, Spinglass
using ProblemReductions: findbest, BruteForce, spinglass_gadget, truth_table, @bit_str, Factoring, reduction_paths, reduceto, extract_solution, target_problem, SpinGlass

@testset "gates" begin
    or_gadget = spinglass_gadget(Val(:∨))
    and_gadget = spinglass_gadget(Val(:∧))
    not_gadget = spinglass_gadget(Val(:¬))

    res = findbest(and_gadget.problem, BruteForce())
    @test sort(res) == [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]]
    tt = truth_table(and_gadget)
    @test length(tt) == 4
    @test tt[bit"00"] == tt[bit"01"] == tt[bit"10"] == bit"0"
    @test tt[bit"11"] == bit"1"

    res = findbest(or_gadget.problem, BruteForce())
    @test sort(res) == [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]]

    res = findbest(not_gadget.problem, BruteForce())
    @test sort(res) == [[0, 1], [1, 0]]
end

@testset "arraymul" begin
    arr = spinglass_gadget(Val(:arraymul))
    tt = truth_table(arr)
    @test length(tt) == 16
    @test tt[bit"0000"] == tt[bit"0001"] == tt[bit"0010"] == bit"00"
    @test tt[bit"0100"] == tt[bit"0101"] == tt[bit"0110"] == tt[bit"0011"] ==
        tt[bit"1000"] == tt[bit"1001"] == tt[bit"1010"]  == bit"10"
    @test tt[bit"0111"] == tt[bit"1011"] ==
        tt[bit"1101"] == tt[bit"1110"] == tt[bit"1100"] == bit"01"
    @test tt[bit"1111"] == bit"11"

    fact = Factoring(2, 3, 15)
    path = reduction_paths(Factoring, SpinGlass)
    spin_glass = reduceto(path[1], fact) |> target_problem
    @test nv(spin_glass.graph) == 63
end