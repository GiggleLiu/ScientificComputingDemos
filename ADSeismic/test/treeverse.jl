using ADSeismic, Test, ForwardDiff
@testset "treeverse" begin
    struct P3{T}
        x::T
        y::T
        z::T
    end

    Base.zero(::Type{P3{T}}) where T = P3(zero(T), zero(T), zero(T))
    Base.zero(::P3{T}) where T = P3(zero(T), zero(T), zero(T))


    @inline function Base.:(+)(a::P3, b::P3)
        P3(a.x + b.x, a.y + b.y, a.z + b.z)
    end

    @inline function Base.:(/)(a::P3, b::Real)
        P3(a.x/b, a.y/b, a.z/b)
    end

    @inline function Base.:(*)(a::Real, b::P3)
        P3(a*b.x, a*b.y, a*b.z)
    end


    function lorentz(t, y, θ)
        P3(10*(y.y-y.x), y.x*(27-y.z)-y.y, y.x*y.y-8/3*y.z)
    end

    function rk4_step(f, t, y, θ; Δt)
        k1 = Δt * f(t, y, θ)
        k2 = Δt * f(t+Δt/2, y + k1 / 2, θ)
        k3 = Δt * f(t+Δt/2, y + k2 / 2, θ)
        k4 = Δt * f(t+Δt, y + k3, θ)
        return y + k1/6 + k2/3 + k3/3 + k4/6
    end

    function rk4(f, y0::T, θ; t0, Δt, Nt) where T
        history = zeros(T, Nt+1)
        history[1] = y0
        y = y0
        for i=1:Nt
            y = rk4_step(f, t0+(i-1)*Δt, y, θ; Δt=Δt)
            @inbounds history[i+1] = y
        end
        return history
    end

    function step_fun(x)
        i_step_fun((0.0, zero(x[2])), x)[1]
    end

    # TODO: use Enzyme.jl to compute the gradient

    @testset "treeverse gradient" begin
        x0 = P3(1.0, 0.0, 0.0)
        for N in [20, 120, 126]
            g_fd = ForwardDiff.gradient(x->rk4(lorentz, P3(x...), nothing; t0=0.0, Δt=3e-3, Nt=N)[end].x, [x0.x, x0.y, x0.z])
            g_tv = treeverse(step_fun, backward, (0.0, x0); δ=4, N=N)
            @test g_fd ≈ [g_tv[2].x, g_tv[2].y, g_tv[2].z]
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

        function getnilanggrad(c::AbstractMatrix{T}) where T
            c = copy(c)
            tu = zeros(T, size(c)..., N+1)
            tφ = zeros(T, size(c)..., N+1)
            tψ = zeros(T, size(c)..., N+1)
            # TODO: use Enzyme.jl to compute the gradient
            res = Enzyme.gradient(loss, (0.0, param, src, srcv, c, tu, tφ, tψ))
            res[end-2], res[end-4], res[end-3]
        end

        s1 = ADSeismic.SeismicState([randn(nx+2,ny+2) for i=1:4]..., Ref(2))
        s4 = ADSeismic.treeverse_step(s1, param, src, srcv, c)
        g_nilang_x, g_nilang_srcv, g_nilang_c = getnilanggrad(copy(c))
        s0 = ADSeismic.SeismicState(Float64, nx, ny)
        gn = ADSeismic.SeismicState(Float64, nx, ny)
        gn.u[45,45] = 1.0
        log = ADSeismic.TreeverseLog()
        res0 = solve(param, src, srcv, copy(c))
        res, (g_tv_x, g_tv_srcv, g_tv_c) = treeverse_solve(s0, x->(gn, zero(srcv), zero(c));
                    param=param, c=copy(c), src,
                    srcv=srcv, δ=50, logger=log)
        @test res.u ≈ res0[:,:,end]
        @test isapprox(g_nilang_srcv, g_tv_srcv)
        @test isapprox(g_nilang_c, g_tv_c)
        @test maximum(g_nilang_c) ≈ maximum(g_tv_c)
        @test g_nilang_x[:,:,2] ≈ g_tv_x.u
    end
end
