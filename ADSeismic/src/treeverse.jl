# Inputs:
# - `s`: The current seismic state containing wavefield values
# - `param`: Acoustic propagator parameters
# - `src`: Source location coordinates
# - `srcv`: Source time function values
# - `c`: Velocity model
# Returns the next seismic state after one time step
function treeverse_step!(s, s2, param, src, srcv, c)
    s2.upre .= s.u
    one_step!(param, s2.u, s.u, s.upre, s2.φ, s2.ψ, param.Σx, param.Σy, c)
    s2.u[src...] += srcv[s2.step[]]*param.DELTAT^2
    return s2
end

# Inputs:
# - `x`: The current seismic state
# - `g`: The gradient of the loss with respect to the next state
# - `param`: Acoustic propagator parameters
# - `src`: Source location coordinates
# - `srcv`: Source time function values
# - `gsrcv`: Gradient with respect to source time function
# - `c`: Velocity model
# - `gc`: Gradient with respect to velocity model
# Returns the gradients with respect to the current state, source time function, and velocity model
function treeverse_grad!(x, g, param, src, srcv, gsrcv, c, gc)
    # TODO: implement this with Enzyme.jl
    # one_step!(param, unext, s.u, s.upre, φ, ψ, param.Σx, param.Σy, c)
end

"""
    treeverse_gradient(s0; param, src, srcv, c, δ=20, logger=TreeverseLog())

* `s0` is the initial state,
"""
function treeverse_gradient(s0, gnf; param, src, srcv, c, δ=20, logger=TreeverseLog())
    f = x->treeverse_step!(x, SeismicState(x.u, copy(x.u), copy(x.φ), copy(x.ψ), Ref(x.step[]+1)), param, src, srcv, c)
    res = []
    function gf(x, g)  # g is a triple of (gx, gsrcv, gc)
        if g === nothing
            y = f(x)
            push!(res, y)
            g = gnf(y)
        end
        gx2, gsrcv2, gc2 = g
        unext, φ, ψ = zero(x.u), copy(x.φ), copy(x.ψ)
        x2 = SeismicState(zero(x.u), unext, φ, ψ, Ref(x.step[]+1))
        gx = SeismicState(zero(x.u), zero(x.u), zero(x.φ), zero(x.ψ), Ref(x.step[]))
        Enzyme.autodiff(Reverse, treeverse_step!, Const, Duplicated(x, gx), Duplicated(x2, gx2), Const(param), Const(src), Duplicated(srcv, gsrcv2), Duplicated(c, gc2))
        return (gx, gsrcv2, gc2)
    end
    g = treeverse(f, gf, copy(s0); δ=δ, N=param.NSTEP-1, f_inplace=true, logger=logger)
    return res[], g
end
