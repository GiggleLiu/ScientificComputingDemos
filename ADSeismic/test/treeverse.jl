using ADSeismic, Test, ForwardDiff, Enzyme

@testset "treeverse" begin
    function lorentz(t, p, θ)
        x, y, z = p
        return [10*(y-x), x*(27-z)-y, x*y-8/3*z]
    end

    function rk4_step(f, t, y, θ; Δt)
        k1 = Δt/6 * f(t, y, θ)
        k2 = Δt/3 * f(t+Δt/2, y .+ k1 ./ 2, θ)
        k3 = Δt/3 * f(t+Δt/2, y .+ k2 ./ 2, θ)
        k4 = Δt/6 * f(t+Δt, y .+ k3, θ)
        return y + k1 + k2 + k3 + k4
    end

    function rk4(f, y0::T, θ; t0, Δt, Nt) where T
        for i=1:Nt
            y0 = rk4_step(f, t0+(i-1)*Δt, y0, θ; Δt=Δt)
        end
        return y0
    end

    # TODO: use Enzyme.jl to compute the gradient
    step_func(x) = rk4_step(lorentz, 0.0, x, nothing; Δt=3e-3)
    function back(x, f_and_g::Nothing)
        # set the gradient of the last state to [1, 0, 0], i.e. differentiate with respect to x
        y = step_func(x)
        return back(x, (y, y, [1, 0, 0.0]))
    end
    function back(x, f_and_g::Tuple)
        function forward(x, y)
            y .= step_func(x)
            return nothing
        end
        x̅ = zero(x)
        result, y, y̅ = f_and_g
        Enzyme.autodiff(Reverse, Const(forward), Duplicated(x, x̅), Duplicated(y, y̅))
        return (result, x, x̅)
    end

    result, g = back(randn(3), (randn(3), randn(3), randn(3)))
    @test g isa Vector{Float64}

    @testset "treeverse gradient" begin
        x0 = [1.0, 0.0, 0.0]
        for N in [20, 120, 126]
            g_fd = ForwardDiff.gradient(x->rk4(lorentz, x, nothing; t0=0.0, Δt=3e-3, Nt=N)[1], x0)
            result_tv, _, g_tv = treeverse(step_func, back, x0; δ=4, N=N)
            @test result_tv ≈ rk4(lorentz, x0, nothing; t0=0.0, Δt=3e-3, Nt=N)
            @test g_fd ≈ g_tv
        end
    end
end

@testset "integrate" begin
    FT = Float64
    h = FT(0.01π)
    dt = FT(0.01)
    α = FT(1e-1)
    function step(src::AbstractArray{T}) where T
        dest = zero(src)
        n = length(dest)
        for i=1:n
            g = α*(src[mod1(i+1, n)] + src[mod1(i-1, n)] - 2*src[i]) / h^2
            dest[i] = src[i] + dt*g
        end
        return dest
    end
    n = 100
    x = zeros(FT, n)
    x[n÷2] = 1
    δ=3
    τ=2
    nsteps = binomial(τ+δ, τ)
    # directsolve
    log = ADSeismic.TreeverseLog()
    g = treeverse(step, (b,c)-> c===nothing ? 1 : c+1, FT.(x); N=nsteps, δ=δ, logger=log)
    @test g == nsteps
    @test log.peak_mem[] == 4
    @test count(x->x.action==:call, log.actions) == 2*nsteps-5
    @test count(x->x.action==:grad, log.actions) == nsteps

    δ=3
    nsteps = 100000
    log = ADSeismic.TreeverseLog()
    g = treeverse(x->x, (b,c)-> c===nothing ? 1 : c+1, 0.0; N=nsteps, δ=δ, logger=log)
    @test log.peak_mem[] == 4
    @test g == nsteps
    δ=9
    g = treeverse(x->x, (b,c)-> c===nothing ? 1 : c+1, 0.0; N=nsteps, δ=δ, logger=log)
    @test log.peak_mem[] == 10
    @test g == nsteps
end

@testset "treeverse gradient" begin
    nx = ny = 50
    N = 1000
    c = 1000*ones(nx+2, ny+2)

    # gradient
    param = AcousticPropagatorParams(nx=size(c,1)-2, ny=size(c,2)-2,
        Rcoef=0.2, dx=20.0, dy=20.0, dt=0.05, nstep=N)
    src = size(c) .÷ 2 .- 1
    srcv = Ricker(param, 100.0, 500.0)

    s1 = ADSeismic.SeismicState([randn(nx+2,ny+2) for i=1:4]..., Ref(2))
    s0 = ADSeismic.SeismicState(Float64, nx, ny)
    gn = ADSeismic.SeismicState(Float64, nx, ny)
    gn.u[45,45] = 1.0
    log = ADSeismic.TreeverseLog()
    res0 = solve(param, src, srcv, copy(c))
    res, (g_tv_x, g_tv_srcv, g_tv_c) = treeverse_gradient(s0, x->(gn, zero(srcv), zero(c));
                param=param, c=copy(c), src,
                srcv=srcv, δ=50, logger=log)
    @test res.u ≈ res0[:,:,end]
end
