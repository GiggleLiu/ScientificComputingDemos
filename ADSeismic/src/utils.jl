export Ricker
"""
    Ricker(epp::Union{ElasticPropagatorParams, AcousticPropagatorParams}, 
    a::Union{PyObject, <:Real}, 
    shift::Union{PyObject, <:Real}, 
    amp::Union{PyObject, <:Real}=1.0)

Returns a Ricker wavelet (a tensor). 
- `epp`: a `ElasticPropagatorParams` or an `AcousticPropagatorParams`
- `a`: Width parameter
- `shift`: Center of the Ricker wavelet
- `amp`: Amplitude of the Ricker wavelet

```math
f(x) = \\mathrm{amp}A (1 - x^2/a^2) exp(-x^2/2 a^2)
```
where 
```math
A = 2/sqrt(3a)pi^1/4
```
"""
function Ricker(epp, 
        a, 
        shift, 
        amp=1.0)
    NT, T = epp.NSTEP, epp.NSTEP*epp.DELTAT
    A = @. 2 / (sqrt(3 * a) * (pi^0.25))
    wsq = @. a^2
    vec =  collect(1:NT) .-shift
    xsq = @. vec^2
    mod = @. (1 - xsq / wsq)
    gauss = @. exp(-xsq / (2 * wsq))
    total = @. amp * A * mod * gauss
    return total
end

struct SafeIndex{T}
    arg::T
end

function SafeIndex(args::Tuple)
    SafeIndex(CartesianIndex(args))
end
SafeIndex(args::Int...) = SafeIndex(args)

function Base.getindex(x::AbstractArray, si::SafeIndex)
    getindex(x, si.arg)
end

function Base.setindex!(x::AbstractArray, val, si::SafeIndex)
    setindex!(x, val, si.arg)
end

