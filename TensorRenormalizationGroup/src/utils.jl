@doc raw"""
    num_grad(f, x; δ=1e-5)

Calculate the numerical gradient of a scalar function `f` at point `x` using
finite differences.

# Arguments
- `f`: A function that takes a scalar and returns a scalar
- `x`: The point at which to evaluate the gradient
- `δ`: The step size for finite differences (default: 1e-5)

# Returns
- The numerical approximation of f'(x)

# Example
```julia
f(x) = x^2
grad = num_grad(f, 3.0)  # Should be approximately 6.0
```
"""
function num_grad(f, x; δ=1e-5)
    return (f(x + δ) - f(x - δ)) / (2 * δ)
end

