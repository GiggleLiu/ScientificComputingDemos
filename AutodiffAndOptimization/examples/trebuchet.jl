# leap frog simulator of a trebuchet
function simulate(x0, v0, θ0, ω0, τ, n)
    x, v, θ, ω = x0, v0, θ0, ω0
    for i in 1:n
        x += τ*v
        v += τ*(-9.8*sin(θ))
        θ += τ*ω
        ω += τ*(9.8/2)*cos(θ)
    end
    return x, v, θ, ω
end
